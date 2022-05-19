package com.example.graduateproject.classfiers

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import com.example.graduateproject.env.Utils
import com.example.graduateproject.env.Utils.loadModelFile
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*

class YoloClassfier :YoloInterfaceClassfier {
    // Config values.
    private val OUTPUT_WIDTH_TINY: IntArray? = intArrayOf(2535, 2535)
    private val OUTPUT_WIDTH_FULL = intArrayOf(10647, 10647)
    private val MASKS_TINY = arrayOf(intArrayOf(3, 4, 5), intArrayOf(1, 2, 3))
    private val ANCHORS_TINY = intArrayOf(
        23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319
    )
    private val XYSCALE_TINY = floatArrayOf(1.05f, 1.05f)

    private val labels = Vector<String>()
    private val NUM_THREADS = 4
    private val isNNAPI = false
    private val isGPU = false
    private var isModelQuantized = false

    // tiny or not
    private val isTiny = true

    // Config values.
    // Pre-allocated buffers.
    private var imgData: ByteBuffer? = null
    lateinit var tfLite: Interpreter
    lateinit var intValues: IntArray
    private val INPUT_SIZE = 416



    @Throws(IOException::class)
    fun create(
        assetManager: AssetManager,
        modelFilename: String,
        labelFilename: String,
        isQuantized: Boolean
    ): YoloClassfier {
        val d=YoloClassfier()
        val actualFilename = labelFilename.split("file:///android_asset/").toTypedArray()[1]
        val labelsInput = assetManager.open(actualFilename)
        val br = BufferedReader(InputStreamReader(labelsInput))
        var line: String?
        while (br.readLine().also { line = it } != null) {
            Log.i("Label Log ",line!!)
            d.labels.add(line)
        }
        br.close()
        try {
            val options = Interpreter.Options()
            options.setNumThreads(NUM_THREADS)
            if (isNNAPI) {
                Log.i("check","1")
                var nnApiDelegate: NnApiDelegate? = null
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    options.setNumThreads(NUM_THREADS)
                    options.setUseNNAPI(false)
                    options.setAllowFp16PrecisionForFp32(true)
                    options.setAllowBufferHandleOutput(true)
                    options.setUseNNAPI(true)
                }
            }
            if (isGPU) {
                Log.i("check","2")
                options.addDelegate(GpuDelegate())
            }
           d.tfLite = Interpreter(loadModelFile(assetManager, modelFilename), options)
            Log.i("check","3")
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
        d.isModelQuantized = isQuantized
        // Pre-allocate buffers.
        val numBytesPerChannel: Int = if (isQuantized) {
            1 // Quantized
        } else {
            4 // Floating point
        }

        d.imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * numBytesPerChannel).order(ByteOrder.nativeOrder())

        d.intValues = IntArray(INPUT_SIZE * INPUT_SIZE)

        return d
    }




    override fun enableStatLogging(debug: Boolean) {

    }

    override fun getStatString(): String? {
        return ""
    }

    override fun close() {

    }

    override fun setNumThreads(num_threads: Int) {
        if (tfLite != null) tfLite!!.setNumThreads(num_threads)
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        if (tfLite != null) tfLite!!.setUseNNAPI(isChecked)
    }

    override fun getObjThresh(): Float {
        return 0.5f
    }


    //non maximum suppression
    protected fun nms(list: ArrayList<YoloInterfaceClassfier.Recognition>): ArrayList<YoloInterfaceClassfier.Recognition>? {
        val nmsList = ArrayList<YoloInterfaceClassfier.Recognition>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq = PriorityQueue(
                50,
                Comparator<YoloInterfaceClassfier.Recognition?> { p0, p1 -> compareValues(p0.confidence,p1.confidence)  })
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a = arrayOfNulls<YoloInterfaceClassfier.Recognition?>(pq.size)
                val detections: Array<YoloInterfaceClassfier.Recognition?> = pq.toArray(a)
                val max = detections[0]
                if (max != null) {
                    nmsList.add(max)
                }
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection = detections[j]
                    val b: RectF? = detection!!.location
                    if (box_iou(max!!.location!!, b!!) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }

    protected fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }

    protected var mNmsThresh = 0.6f
    protected fun box_intersection(a: RectF, b: RectF): Float {
        val w: Float = overlap(
            (a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left
        )
        val h: Float = overlap(
            (a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0.0F else w * h
    }

    protected fun box_union(a: RectF, b: RectF): Float {
        val i: Float = box_intersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }
    protected fun box_iou(a: RectF, b: RectF): Float {
        return box_intersection(a, b) / box_union(a, b)
    }

    protected val BATCH_SIZE = 1
    protected val PIXEL_SIZE = 3

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    protected fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer? {
        val byteBuffer =
            ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues =
            IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat((`val` shr 16 and 0xFF) / 255.0f)
                byteBuffer.putFloat((`val` shr 8 and 0xFF) / 255.0f)
                byteBuffer.putFloat((`val` and 0xFF) / 255.0f)
            }
        }
        return byteBuffer
    }


    private fun getDetectionsForFull(
        byteBuffer: ByteBuffer,
        bitmap: Bitmap
    ): ArrayList<YoloInterfaceClassfier.Recognition> {
        val detections = ArrayList<YoloInterfaceClassfier.Recognition>()
        val outputMap: MutableMap<Int, Any> = HashMap()
        outputMap[0] = Array(1) {
            Array(
                OUTPUT_WIDTH_FULL[0]
            ) {
                FloatArray(
                    4
                )
            }
        }
        outputMap[1] = Array(1) {
            Array(
                OUTPUT_WIDTH_FULL[1]
            ) {
                FloatArray(
                    labels.size
                )
            }
        }
        val inputArray = arrayOf<Any>(byteBuffer)
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)
        val gridWidth: Int = OUTPUT_WIDTH_FULL[0]
        val bboxes = outputMap[0] as Array<Array<FloatArray>>?
        val out_score = outputMap[1] as Array<Array<FloatArray>>?
        for (i in 0 until gridWidth) {
            var maxClass = 0f
            var detectedClass = -1
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = out_score!![0][i][c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val score = maxClass
            if (score > getObjThresh()) {
                val xPos = bboxes!![0][i][0]
                val yPos = bboxes[0][i][1]
                val w = bboxes[0][i][2]
                val h = bboxes[0][i][3]
                val rectF = RectF(
                    Math.max(0f, xPos - w / 2),
                    Math.max(0f, yPos - h / 2),
                    Math.min((bitmap.width - 1).toFloat(), xPos + w / 2),
                    Math.min((bitmap.height - 1).toFloat(), yPos + h / 2)
                )
                detections.add(
                    YoloInterfaceClassfier.Recognition(
                        "" + i,
                        labels[detectedClass],
                        score,
                        rectF,
                        detectedClass
                    )
                )
            }
        }
        return detections
    }

    private fun getDetectionsForTiny(
        byteBuffer: ByteBuffer,
        bitmap: Bitmap
    ): ArrayList<YoloInterfaceClassfier.Recognition> {
        val detections = ArrayList<YoloInterfaceClassfier.Recognition>()
        val outputMap: MutableMap<Int, Any> = HashMap()
        outputMap[0] = Array(1) {
            Array(
                OUTPUT_WIDTH_TINY!![0]
            ) {
                FloatArray(
                    4
                )
            }
        }
        outputMap[1] = Array(1) {
            Array(
                OUTPUT_WIDTH_TINY!![1]
            ) {
                FloatArray(
                    labels.size
                )
            }
        }
        val inputArray = arrayOf<Any>(byteBuffer)
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap)
        val gridWidth: Int = OUTPUT_WIDTH_TINY!![0]
        val bboxes = outputMap[0] as Array<Array<FloatArray>>?
        val out_score = outputMap[1] as Array<Array<FloatArray>>?
        for (i in 0 until gridWidth) {
            var maxClass = 0f
            var detectedClass = -1
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = out_score!![0][i][c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val score = maxClass
            if (score > getObjThresh()) {
                val xPos = bboxes!![0][i][0]
                val yPos = bboxes[0][i][1]
                val w = bboxes[0][i][2]
                val h = bboxes[0][i][3]
                val rectF = RectF(
                    Math.max(0f, xPos - w / 2),
                    Math.max(0f, yPos - h / 2),
                    Math.min((bitmap.width - 1).toFloat(), xPos + w / 2),
                    Math.min((bitmap.height - 1).toFloat(), yPos + h / 2)
                )
                detections.add(
                    YoloInterfaceClassfier.Recognition(
                        "" + i,
                        labels[detectedClass],
                        score,
                        rectF,
                        detectedClass
                    )
                )
            }
        }
        return detections
    }

    override fun recognizeImage(bitmap: Bitmap?): List<YoloInterfaceClassfier.Recognition>? {
        val byteBuffer = convertBitmapToByteBuffer(bitmap!!)

        val detections: ArrayList<YoloInterfaceClassfier.Recognition>
        if (isTiny) {
            detections = getDetectionsForTiny(byteBuffer!!, bitmap)
        } else {
            detections = getDetectionsForFull(byteBuffer!!, bitmap)
        }
        return nms(detections)
    }



}