package com.example.graduateproject


import android.graphics.*
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.graduateproject.classfiers.YoloClassfier
import com.example.graduateproject.classfiers.YoloInterfaceClassfier
import com.example.graduateproject.databinding.ActivityMainBinding
import com.example.graduateproject.env.ImageUtils
import com.example.graduateproject.env.Utils
import com.example.graduateproject.tracker.MultiBoxTracker
import java.io.IOException
import java.util.*

class MainActivity : AppCompatActivity() {
    val MINIMUM_CONFIDENCE_TF_OD_API=0.5f
    lateinit var bitmap:Bitmap

    //제꺼보고 참고하실때, classfiers, customView, env, tracker가 있어야 YOLO 객체탐지 기능이 작동합니다.


    protected var previewWidth = 0
    protected var previewHeight = 0
    private var detector: YoloClassfier? = null
    lateinit var binding:ActivityMainBinding


    @RequiresApi(Build.VERSION_CODES.P)
    override fun onCreate(savedInstanceState: Bundle?) {
        binding= ActivityMainBinding.inflate(layoutInflater)
        super.onCreate(savedInstanceState)
        setContentView(binding.root)


        val getContent = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            // 이미지의 uri값을 활용해 main activity의 이미지 뷰 변경
            binding.imageView.setImageURI(uri)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Toast.makeText(this,"Bitmap 얻어오기",Toast.LENGTH_SHORT).show()
                bitmap=
                    Utils.processBitmap(ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver,uri!!),
                    ImageDecoder.OnHeaderDecodedListener { decoder, info, source ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = true
                }),TF_OD_API_INPUT_SIZE)


            }

            //sdk 버전이 낮으면 아래를 쓸 수 있는데 윗경우를 권장합니다.
            else{
                Toast.makeText(this,"Bitmap 얻어오기2",Toast.LENGTH_SHORT).show()
                bitmap= MediaStore.Images.Media.getBitmap(contentResolver,uri)
            }
        }


        //여기서는 갤러리와 카메라에서 사진을 가져오는 법을 서술하려고 합니다.
        binding.button.setOnClickListener {
            //토스트 메세지 출력
            Toast.makeText(this,"사진 변경",Toast.LENGTH_SHORT).show()
            //getContent를 통해 갤러리 접근, getContent의 registerForActivityResult 실행
            getContent.launch("image/*")

        }


        binding.button2.setOnClickListener {
            //의상 탐지 버튼인데, 비트맵(사진)을 어디에 넣어야할지 알 수 있을 겁니다.
            Toast.makeText(this,"의상 탐지",Toast.LENGTH_SHORT).show()


            val handler = Handler(Looper.getMainLooper())

            Thread {
                //의상 탐지를 위해 handler를 위처럼 적어주시고, detector에 bitmap이미지를 넣되, 오류가 나면 49 line에 있는 listener를 bitmap에 쓰셔야합니다.
                val results: List<YoloInterfaceClassfier.Recognition>? =
                    detector?.recognizeImage(bitmap)

                handler.post(Runnable {
                    //handleresult가 옷의 위치를 그리는 함수입니다. handresult를 변형하셔서, 위치를 가져오는게 좋아요.
                    if (results != null) {
                        handleResult(bitmap, results)
                    }
                })
            }.start()

        }
        val source = Utils.getBitmapFromAsset(this@MainActivity, "267.jpg")
        bitmap= source?.let { Utils.processBitmap(it,TF_OD_API_INPUT_SIZE) }!!
        binding.imageView.setImageBitmap(source.let { Utils.processBitmap(it,TF_OD_API_INPUT_SIZE) })
        initBox()
    }



    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    lateinit var tracker: MultiBoxTracker
    val TF_OD_API_INPUT_SIZE = 416

    private val MAINTAIN_ASPECT = false
    private val sensorOrientation = 90

    private val TF_OD_API_IS_QUANTIZED = false

    //나중에 모델 더 좋게 학습하면 모델이름을 바꾸거나 업데이트하겠죠.
    private val TF_OD_API_MODEL_FILE = "yolov4_1.tflite"

    private val TF_OD_API_LABELS_FILE = "file:///android_asset/obj.txt"

    private fun initBox() {
        previewHeight = 416
        previewWidth = 416
        frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
            sensorOrientation, MAINTAIN_ASPECT
        )

        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        tracker = MultiBoxTracker(this)

        binding.trackingOverlay.addCallback { canvas: Canvas -> tracker.draw(canvas) }

        tracker.setFrameConfiguration(
            TF_OD_API_INPUT_SIZE,
            TF_OD_API_INPUT_SIZE,
            sensorOrientation
        )
        try {
            Log.i("main fail :","check")
            detector = YoloClassfier().create(
                assets,
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_IS_QUANTIZED
            )

        } catch (e: IOException) {
            e.printStackTrace()
            Log.i("main fail :","Exception initializing classifier!")
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
    }


    private fun handleResult(bitmap: Bitmap, results: List<YoloInterfaceClassfier.Recognition>) {
        val canvas = Canvas(bitmap)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        val mappedRecognitions: List<YoloInterfaceClassfier.Recognition> = LinkedList<YoloInterfaceClassfier.Recognition>()
        for (result in results) {

            val location: RectF? = result.location
            if (location != null && result.confidence!! >= MINIMUM_CONFIDENCE_TF_OD_API) {
                //여기있는 location이 옷이 있는 좌표입니다. 그것을 활용해서 옷의 위치를 추출해주시면 될 것 같습니다

                Log.i("results",location.toShortString())
                canvas.drawRect(location, paint)
            }
        }
        binding.imageView.setImageBitmap(bitmap)
    }



}