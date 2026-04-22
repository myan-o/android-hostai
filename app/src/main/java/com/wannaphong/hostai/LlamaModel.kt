package com.wannaphong.hostai

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock as mutexWithLock
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Data class to hold all generation/completion parameters.
 * These parameters are compatible with LiteRT's SamplerConfig.
 */
data class GenerationConfig(
    val maxTokens: Int = 100,
    val temperature: Double = 0.7,
    val topK: Int = 40,
    val topP: Double = 0.95,
    val seed: Int = -1,
    val chatTemplateKwArgs: Map<String, Any> = emptyMap()
)

/**
 * LLM model interface using LiteRT (LLM) library.
 * 
 * This implementation uses the LiteRT library which provides
 * native LLM inference optimized for Android/ARM devices with GPU acceleration.
 */
class LlamaModel(
    private val contentResolver: ContentResolver, 
    private val context: Context
) {
    private var modelName = "litert-model"
    private var modelPath: String? = null
    @Volatile private var isLoaded = false
    
    // LiteRT components
    @Volatile private var engine: Engine? = null
    @Volatile private var _conversation: Conversation? = null
    private var currentSessionId: String? = null
    private var currentFullHistory: List<Message>? = null
    private val scope = CoroutineScope(Dispatchers.IO)

    // Global Mutex to serialise the entire conversation lifecycle.
    // LiteRT supports only one active session at a time: a second
    // createConversation() call while a session is open throws
    // FAILED_PRECONDITION.  This mutex is held from createConversation()
    // until the conversation is closed, ensuring only one request uses
    // the engine at a time even when concurrency > 1 is configured.
    private val inferenceMutex = Mutex()

    // Read/Write lock to guard the engine lifecycle.
    // generate*() methods acquire the read lock (and inferenceMutex for
    // conversation serialisation).
    // close() acquires the write lock, which blocks until every in-flight
    // native sendMessage() call has finished – preventing a native crash where
    // the engine is freed while it is still executing inference.
    private val engineLifecycleLock = ReentrantReadWriteLock()

    // Cache SettingsManager to avoid repeated instantiation
    private val settingsManager by lazy { SettingsManager(context) }
    
    companion object {
        private const val TAG = "LlamaModel"
        private const val DEFAULT_MAX_TOKENS = 2048
    }
    
    fun loadModel(modelPath: String): Boolean {
        this.modelPath = modelPath
        
        LogManager.i(TAG, "Loading model from path: $modelPath")
        
        // Handle different path types
        if (modelPath == "mock-model") {
            // For mock model, just mark as loaded
            LogManager.i(TAG, "Using mock model")
            isLoaded = true
            return true
        }

        val enginePath: String
        if (modelPath.startsWith("content://")) {
            // LiteRT's native engine requires a real file-system path with the
            // correct file extension – it cannot follow /proc/self/fd/<n> symlinks.
            // Copy the model from the content URI to the app's internal model cache
            // directory, keeping the original filename (and therefore the .litertlm
            // extension).  Subsequent starts reuse the cached copy so no extra I/O
            // is needed after the first load.
            val uri = Uri.parse(modelPath)
            val fileName = getFileNameFromUri(uri)
                ?: uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':')
                ?: "model.litertlm"
            modelName = fileName

            val fileSize = getFileSizeFromUri(uri)

            val cachedFile = getCachedModelFile(fileName, fileSize)
            enginePath = if (cachedFile != null) {
                LogManager.i(TAG, "Using cached model file: ${cachedFile.absolutePath}")
                cachedFile.absolutePath
            } else {
                val sizeDisplay = if (fileSize > 0) "${fileSize / 1024 / 1024} MB" else "unknown size"
                LogManager.i(TAG, "Copying model from URI to internal cache ($sizeDisplay)…")
                val destFile = File(getModelCacheDir(), fileName)
                try {
                    contentResolver.openInputStream(uri)?.use { input ->
                        destFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    } ?: run {
                        LogManager.e(TAG, "Failed to open input stream for URI: $modelPath")
                        return false
                    }
                } catch (e: Exception) {
                    LogManager.e(TAG, "Failed to copy model from URI: ${e.message}", e)
                    destFile.delete()
                    return false
                }
                LogManager.i(TAG, "Model cached at: ${destFile.absolutePath}")
                destFile.absolutePath
            }
        } else {
            // It's a plain file path
            val file = File(modelPath)
            if (file.exists()) {
                modelName = file.name
                LogManager.i(TAG, "Model file found: $modelName (${file.length() / 1024 / 1024} MB)")
            } else {
                LogManager.e(TAG, "Model file not found at path: $modelPath")
                return false
            }
            enginePath = modelPath
        }
        return loadFromPath(enginePath)
    }

    /** Returns the cache directory used to store model copies from content URIs. */
    private fun getModelCacheDir(): File {
        val dir = File(context.filesDir, "model_cache")
        if (!dir.exists() && !dir.mkdirs()) {
            LogManager.w(TAG, "Failed to create model cache directory: ${dir.absolutePath}")
        }
        return dir
    }

    /**
     * Returns the display filename reported by ContentResolver for [uri],
     * or null if the query fails.
     */
    private fun getFileNameFromUri(uri: Uri): String? {
        return try {
            contentResolver.query(
                uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Returns the file size (bytes) reported by ContentResolver for [uri],
     * or -1 if unknown.
     */
    private fun getFileSizeFromUri(uri: Uri): Long {
        return try {
            contentResolver.query(
                uri, arrayOf(OpenableColumns.SIZE), null, null, null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getLong(0) else -1L
            } ?: -1L
        } catch (e: Exception) {
            -1L
        }
    }

    /**
     * Returns the cached file if it exists, has the right name, and (when
     * [expectedSize] is positive) its size matches.  Returns null otherwise.
     */
    private fun getCachedModelFile(fileName: String, expectedSize: Long): File? {
        val file = File(getModelCacheDir(), fileName)
        if (!file.exists()) return null
        if (expectedSize > 0 && file.length() != expectedSize) return null
        return file
    }

    /**
     * Initialise the LiteRT engine from a real file-system path.
     */
    private fun loadFromPath(enginePath: String): Boolean {
        return try {
            LogManager.i(TAG, "Initializing LiteRT with model: $modelName")

            // Get backend preference from settings
            val backend = when (settingsManager.getBackend()) {
                SettingsManager.BACKEND_NPU -> {
                    LogManager.i(TAG, "Using NPU backend for inference")
                    Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
                }
                SettingsManager.BACKEND_GPU -> {
                    LogManager.i(TAG, "Using GPU backend for inference")
                    Backend.GPU()
                }
                else -> {
                    LogManager.i(TAG, "Using CPU backend for inference")
                    Backend.CPU()
                }
            }

            // Get max context length from settings
            val maxContextLength = settingsManager.getMaxContextLength()
            LogManager.i(TAG, "Using max context length: $maxContextLength tokens")

            // Compiled-kernel cache directory: speeds up subsequent model loads by reusing
            // pre-compiled GPU/NPU kernels instead of recompiling them on every launch.
            val cacheDirFile = File(context.cacheDir, "litert_cache")
            if (!cacheDirFile.exists() && !cacheDirFile.mkdirs()) {
                LogManager.w(TAG, "Failed to create LiteRT cache directory; compiled kernels will not be cached")
            }
            val cacheDir = cacheDirFile.absolutePath

            // Create engine config with selected backend.
            // Only add vision/audio backends for multimodal models (e.g. Gemma-3N).
            // Text-only models fail with "Unsupported or unknown file format" when
            // these backends are specified.
            val useMultimodal = settingsManager.isMultimodalEnabled()
            val engineConfig = if (useMultimodal) {
                LogManager.i(TAG, "Multimodal mode enabled: adding vision (GPU) and audio (CPU) backends")
                EngineConfig(
                    modelPath = enginePath,
                    backend = backend,
                    maxNumTokens = maxContextLength,
                    cacheDir = cacheDir,
                    visionBackend = Backend.GPU(),
                    audioBackend = Backend.CPU()
                )
            } else {
                EngineConfig(
                    modelPath = enginePath,
                    backend = backend,
                    maxNumTokens = maxContextLength,
                    cacheDir = cacheDir
                )
            }
            
            // Initialize engine (this can take time, already on IO thread)
            val newEngine = Engine(engineConfig)
            newEngine.initialize()
            
            // Only set engine and isLoaded if initialization succeeds
            engine = newEngine
            isLoaded = true
            
            LogManager.i(TAG, "LiteRT engine initialized successfully with ${settingsManager.getBackend().uppercase()} backend")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            LogManager.e(TAG, "Failed to load model: ${e.message}", e)
            engine = null
            isLoaded = false
            false
        }
    }
    
    fun isModelLoaded(): Boolean {
        return isLoaded
    }
    
    fun getModelName(): String = modelName
    
    fun getModelPath(): String? = modelPath

    /**
     * Create a new conversation for a single request.
     * A fresh conversation is created for every request and closed after use,
     * preventing stale state from causing failures on subsequent requests.
     *
     * **Must be called while [inferenceMutex] is held** – the LiteRT engine
     * only supports one live session at a time.
     *
     * @param config Sampler configuration for the conversation
     * @return The conversation instance, or null if creation fails
     */
    private fun createConversation(config: GenerationConfig, sessionId: String, history: List<Message>): Conversation? {
        if (_conversation != null && currentSessionId == sessionId && currentFullHistory == history) {
            return _conversation
        }

        _conversation?.close()
        currentSessionId = sessionId
        currentFullHistory = history

        _conversation = try {
            val currentEngine = engine
                ?: throw IllegalStateException("Engine is not initialized")

            val samplerConfig = SamplerConfig(
                topK = config.topK,
                topP = config.topP,
                temperature = config.temperature
            )

            val conversationConfig = ConversationConfig(
                systemInstruction = null,
                initialMessages = history,
                samplerConfig = samplerConfig
            )

            currentEngine.createConversation(conversationConfig)
        } catch (e: Exception) {
            LogManager.e(TAG, "Failed to create conversation: ${e.message}")
            null
        }
        return _conversation
    }

    /**
     * Generate text with full configuration support.
     * @param prompt The input prompt text
     * @param config Generation configuration with all parameters (optional)
     * @param sessionId Unused – kept for API compatibility
     * @return Generated text
     */
    fun generate(prompt: String, config: GenerationConfig = GenerationConfig(), sessionId: String = "", history: List<Message> = emptyList()): String {
        if (!isModelLoaded()) {
            val errorMsg = "Error: Model not loaded. Please load a model first."
            LogManager.e(TAG, errorMsg)
            return errorMsg
        }

        LogManager.i(TAG, "Generating response with prompt (length: ${prompt.length})")
        LogManager.d(TAG, "Config: maxTokens=${config.maxTokens}, temp=${config.temperature}, topK=${config.topK}, topP=${config.topP}")
        
        // For mock model, return a simple response
        if (modelPath == "mock-model") {
            val promptPreview = if (prompt.length > 50) prompt.take(50) + "..." else prompt
            return "This is a mock response from the model. In production, this would be the actual LLM output for prompt: \"$promptPreview\""
        }

        // runBlocking bridges this non-suspend function (called by the Javalin HTTP handler)
        // with the suspend world. Javalin handlers run on dedicated IO threads so blocking is acceptable.
        // The read lock prevents the engine from being closed (write lock in close()) while
        // sendMessage() is executing in native code.
        // inferenceMutex is held for the full conversation lifetime so that the next request
        // cannot call createConversation() before this one has called conversation.close().
        return engineLifecycleLock.read {
            // Re-check inside the lock: close() sets isLoaded = false while holding the write
            // lock, so if we reach here after close() completed, we see the updated value.
            if (!isLoaded) {
                return@read "Error: Model not loaded. Please load a model first."
            }
            runBlocking {
                inferenceMutex.mutexWithLock {
                    var conversation: Conversation? = null
                    try {
                        conversation = createConversation(config, sessionId, history)

                        if (conversation == null) {
                            val errorMsg = "Error: Failed to create conversation"
                            LogManager.e(TAG, errorMsg)
                            return@mutexWithLock errorMsg
                        }

                        // Send message and get response synchronously
                        val userMessage = Message.user(prompt)
                        val response = conversation.sendMessage(userMessage, config.chatTemplateKwArgs)
                        val result = response.toString()
                        currentFullHistory = (currentFullHistory ?: emptyList()) + userMessage + Message.model(result)
                        LogManager.i(TAG, "Generation completed successfully (length: ${result.length})")
                        result
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to generate response", e)
                        LogManager.e(TAG, "Failed to generate response: ${e.message}", e)
                        "Error: ${e.message}"
                    } finally {
                    }
                }
            }
        }
    }

    /**
     * Generate text with multimodal content support (images, audio).
     * @param messages List of Message objects
     * @param config Generation configuration with all parameters (optional)
     * @param sessionId Unused – kept for API compatibility
     * @return Generated text
     */
    fun generateWithContents(messages: List<Message>, config: GenerationConfig = GenerationConfig(), sessionId: String = ""): String {
        if (!isModelLoaded()) {
            val errorMsg = "Error: Model not loaded. Please load a model first."
            LogManager.e(TAG, errorMsg)
            return errorMsg
        }

        LogManager.i(TAG, "Generating multimodal response with ${messages.size} messages")
        LogManager.d(TAG, "Config: maxTokens=${config.maxTokens}, temp=${config.temperature}, topK=${config.topK}, topP=${config.topP}")

        if (modelPath == "mock-model") {
            return "This is a mock multimodal response from the model with ${messages.size} messages."
        }

        return engineLifecycleLock.read {
            if (!isLoaded) {
                return@read "Error: Model not loaded. Please load a model first."
            }
            runBlocking {
                inferenceMutex.mutexWithLock {
                    var conversation: Conversation? = null
                    try {
                        val history = messages.dropLast(1)
                        val userMessage = messages.last()

                        conversation = createConversation(config, sessionId, history)

                        if (conversation == null) {
                            val errorMsg = "Error: Failed to create conversation"
                            LogManager.e(TAG, errorMsg)
                            return@mutexWithLock errorMsg
                        }

                        val response = conversation.sendMessage(userMessage, config.chatTemplateKwArgs)
                        val result = response.toString()
                        currentFullHistory = (currentFullHistory ?: emptyList()) + userMessage + Message.model(result)
                        LogManager.i(TAG, "Multimodal generation completed successfully (length: ${result.length})")
                        result
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to generate multimodal response", e)
                        LogManager.e(TAG, "Failed to generate multimodal response: ${e.message}", e)
                        "Error: ${e.message}"
                    } finally {
                    }
                }
            }
        }
    }

    /**
     * Legacy method for backward compatibility.
     * @deprecated Use generate(prompt, GenerationConfig) instead
     */
    @Deprecated("Use generate(prompt, GenerationConfig) for full parameter control")
    fun generate(prompt: String, maxTokens: Int = 100, temperature: Float = 0.7f): String {
        return generate(prompt, GenerationConfig(maxTokens = maxTokens, temperature = temperature.toDouble()))
    }

    /**
     * Generate text with streaming and full configuration support.
     * @param prompt The input prompt text
     * @param config Generation configuration with all parameters (optional)
     * @param sessionId Unused – kept for API compatibility
     * @param onToken Callback for each generated token
     * @return Job that can be cancelled, or null on error
     */
    fun generateStream(
        prompt: String,
        config: GenerationConfig = GenerationConfig(),
        sessionId: String = "",
        history: List<Message> = emptyList(),
        onToken: (String) -> Unit
    ): Job? {
        if (!isModelLoaded()) {
            onToken("Error: Model not loaded. Please load a model first.")
            return null
        }

        LogManager.d(TAG, "Streaming - config: maxTokens=${config.maxTokens}, temp=${config.temperature}, topK=${config.topK}, topP=${config.topP}")

        // For mock model, simulate streaming
        if (modelPath == "mock-model") {
            return scope.launch {
                val mockResponse = "This is a mock streaming response from the model. "
                onToken(mockResponse)
            }
        }

        return scope.launch {
            // Hold the read lock for the entire streaming session so that close()
            // (which takes the write lock) cannot free the native engine while
            // sendMessageAsync callbacks are still firing.
            engineLifecycleLock.read {
                // Re-check inside the lock (same pattern as generate()).
                if (!isLoaded) {
                    onToken("Error: Model not loaded. Please load a model first.")
                    return@read
                }
                inferenceMutex.mutexWithLock {
                    var conversation: Conversation? = null
                    try {
                        conversation = createConversation(config, sessionId, history)

                        if (conversation == null) {
                            LogManager.e(TAG, "Failed to create conversation")
                            onToken("Error: Failed to create conversation")
                            return@mutexWithLock
                        }

                        val userMessage = Message.user(prompt)
                        val fullResponse = StringBuilder()
                        suspendCancellableCoroutine<Unit> { continuation ->
                            val resumed = AtomicBoolean(false)
                            val in_think = AtomicBoolean(false)

                            val callback = object : MessageCallback {
                                override fun onMessage(message: Message) {
                                    // Emit each token chunk directly as it arrives from the engine.
                                    // No buffering or artificial delays — let the native engine pace output.
                                    // Wrap in try-catch: exceptions must never escape a JNI callback or
                                    // they will crash the native engine / the Android process.
                                    try {
                                        var token = ""
                                        if (in_think.get()) {
                                            if (null == message.channels["thought"]) {
                                                in_think.set(false)
                                                token = "\n</think>\n" + message.toString()
                                                fullResponse.append(message.toString())
                                            }
                                            else {
                                                token = message.channels["thought"] ?: ""
                                            }
                                        }
                                        else {
                                            if (null != message.channels["thought"]) {
                                                in_think.set(true)
                                                token = "<think>\n" + message.channels["thought"]
                                            }
                                            else {
                                                token = message.toString()
                                                fullResponse.append(message.toString())
                                            }
                                        }
                                        fullResponse.append(token)
                                        onToken(token)
                                    } catch (e: Exception) {
                                        LogManager.w(TAG, "Token callback error (client may have disconnected): ${e.message}")
                                        if (resumed.compareAndSet(false, true)) {
                                            continuation.resumeWithException(e)
                                        }
                                    }
                                }

                                override fun onDone() {
                                    LogManager.i(TAG, "Streaming completed")
                                    if (resumed.compareAndSet(false, true)) {
                                        continuation.resume(Unit)
                                    }
                                }

                                override fun onError(throwable: Throwable) {
                                    Log.e(TAG, "Streaming error", throwable)
                                    LogManager.e(TAG, "Streaming error: ${throwable.message}", throwable)
                                    if (resumed.compareAndSet(false, true)) {
                                        continuation.resumeWithException(throwable)
                                    }
                                }
                            }

                            conversation.sendMessageAsync(userMessage, callback, config.chatTemplateKwArgs)
                        }
                        currentFullHistory = (currentFullHistory ?: emptyList()) + userMessage + Message.model(fullResponse.toString())
                    } catch (e: Exception) {
                        Log.e(TAG, "Streaming failed", e)
                        LogManager.e(TAG, "Streaming failed: ${e.message}", e)
                        try { onToken("Error: ${e.message}") } catch (ignored: Exception) {
                            // Client may have already disconnected; nothing to do.
                        }
                    } finally {
                    }
                }
            }
        }
    }

    /**
     * Generate text with streaming and multimodal content support (images, audio).
     * @param messages List of Message objects
     * @param config Generation configuration with all parameters (optional)
     * @param sessionId Unused – kept for API compatibility
     * @param onToken Callback for each generated token
     * @return Job that can be cancelled, or null on error
     */
    fun generateStreamWithContents(
        messages: List<Message>,
        config: GenerationConfig = GenerationConfig(),
        sessionId: String = "",
        onToken: (String) -> Unit
    ): Job? {
        if (!isModelLoaded()) {
            onToken("Error: Model not loaded. Please load a model first.")
            return null
        }

        LogManager.d(TAG, "Streaming multimodal with ${messages.size} messages - config: maxTokens=${config.maxTokens}, temp=${config.temperature}, topK=${config.topK}, topP=${config.topP}")

        // For mock model, simulate streaming
        if (modelPath == "mock-model") {
            return scope.launch {
                val mockResponse = "This is a mock multimodal streaming response from the model with ${messages.size} messages. "
                onToken(mockResponse)
            }
        }

        return scope.launch {
            // Hold the read lock for the entire streaming session so that close()
            // (which takes the write lock) cannot free the native engine while
            // sendMessageAsync callbacks are still firing.
            engineLifecycleLock.read {
                // Re-check inside the lock (same pattern as generateWithContents()).
                if (!isLoaded) {
                    onToken("Error: Model not loaded. Please load a model first.")
                    return@read
                }
                inferenceMutex.mutexWithLock {
                    var conversation: Conversation? = null
                    try {
                        val history = if (messages.size > 1) messages.subList(0, messages.size - 1) else emptyList()
                        val lastMessage = messages.last()
                        conversation = createConversation(config, sessionId, history)

                        if (conversation == null) {
                            LogManager.e(TAG, "Failed to create conversation")
                            onToken("Error: Failed to create conversation")
                            return@mutexWithLock
                        }

                        val fullResponse = StringBuilder()
                        suspendCancellableCoroutine<Unit> { continuation ->
                            val resumed = AtomicBoolean(false)
                            val in_think = AtomicBoolean(false)

                            val callback = object : MessageCallback {
                                override fun onMessage(message: Message) {
                                    // Emit each token chunk directly as it arrives from the engine.
                                    // Wrap in try-catch: exceptions must never escape a JNI callback or
                                    // they will crash the native engine / the Android process.
                                    try {
                                        var token = ""
                                        if (in_think.get()) {
                                            if (null == message.channels["thought"]) {
                                                in_think.set(false)
                                                token = "\n</think>\n" + message.toString()
                                                fullResponse.append(message.toString())
                                            }
                                            else {
                                                token = message.channels["thought"] ?: ""
                                            }
                                        }
                                        else {
                                            if (null != message.channels["thought"]) {
                                                in_think.set(true)
                                                token = "<think>\n" + message.channels["thought"]
                                            }
                                            else {
                                                token = message.toString()
                                                fullResponse.append(message.toString())
                                            }
                                        }
                                        onToken(token)
                                    } catch (e: Exception) {
                                        LogManager.w(TAG, "Multimodal token callback error (client may have disconnected): ${e.message}")
                                        if (resumed.compareAndSet(false, true)) {
                                            continuation.resumeWithException(e)
                                        }
                                    }
                                }

                                override fun onDone() {
                                    LogManager.i(TAG, "Multimodal streaming completed")
                                    if (resumed.compareAndSet(false, true)) {
                                        continuation.resume(Unit)
                                    }
                                }

                                override fun onError(throwable: Throwable) {
                                    Log.e(TAG, "Multimodal streaming error", throwable)
                                    LogManager.e(TAG, "Multimodal streaming error: ${throwable.message}", throwable)
                                    if (resumed.compareAndSet(false, true)) {
                                        continuation.resumeWithException(throwable)
                                    }
                                }
                            }

                            conversation.sendMessageAsync(lastMessage, callback, config.chatTemplateKwArgs)
                        }
                        currentFullHistory = (currentFullHistory ?: emptyList()) + lastMessage + Message.model(fullResponse.toString())
                    } catch (e: Exception) {
                        Log.e(TAG, "Multimodal streaming failed", e)
                        LogManager.e(TAG, "Multimodal streaming failed: ${e.message}", e)
                        try { onToken("Error: ${e.message}") } catch (ignored: Exception) {
                            // Client may have already disconnected; nothing to do.
                        }
                    } finally {
                    }
                }
            }
        }
    }

    /**
     * Legacy method for backward compatibility.
     * @deprecated Use generateStream(prompt, GenerationConfig, onToken) instead
     */
    @Deprecated("Use generateStream(prompt, GenerationConfig, onToken) for full parameter control")
    fun generateStream(
        prompt: String,
        maxTokens: Int = 100,
        temperature: Float = 0.7f,
        onToken: (String) -> Unit
    ): Job? {
        return generateStream(prompt, GenerationConfig(maxTokens = maxTokens, temperature = temperature.toDouble()), "", emptyList(), onToken)
    }

    /**
     * Cleanup resources, optionally closing the engine.
     * Must be called while holding engineLifecycleLock (write lock) when closeEngine is true.
     */
    private fun cleanup(closeEngine: Boolean = false) {
        try {
            if (closeEngine) {
                // Cancel streaming coroutines BEFORE closing the native engine so that
                // any in-flight sendMessageAsync callbacks see a cancelled scope and do
                // not attempt to use engine resources after they are freed.
                scope.cancel()
                engine?.close()
                engine = null
            }

            isLoaded = false
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
            LogManager.e(TAG, "Error during cleanup: ${e.message}", e)
        }
    }
    
    fun unload() {
        LogManager.i(TAG, "Unloading model")
        cleanup(closeEngine = false)
        modelPath = null
    }
    
    /**
     * Explicitly release native resources.
     * Call this when you're done with the model to free memory immediately.
     *
     * Acquires the write lock which blocks until every in-flight generate() call
     * (holding the read lock) has returned.  This prevents the engine from being
     * freed while a native sendMessage() is still executing and causing a crash.
     * isLoaded is set to false inside the write lock so that any thread that
     * races through isModelLoaded() and then waits on the read lock will see
     * the updated flag and bail out without touching a closed engine.
     */
    fun close() {
        LogManager.i(TAG, "Closing model and releasing resources")
        // The write lock waits for all current read-lock holders (in-flight
        // generate / generateWithContents calls) to complete before proceeding.
        engineLifecycleLock.write {
            isLoaded = false
            cleanup(closeEngine = true)
        }
    }
}
