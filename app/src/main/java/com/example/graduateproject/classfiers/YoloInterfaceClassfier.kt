package com.example.graduateproject.classfiers

import android.graphics.Bitmap
import android.graphics.RectF

interface YoloInterfaceClassfier {
    fun recognizeImage(bitmap: Bitmap?): List<Recognition>?

    fun enableStatLogging(debug: Boolean)

    fun getStatString(): String?

    fun close()

    fun setNumThreads(num_threads: Int)

    fun setUseNNAPI(isChecked: Boolean)

    fun getObjThresh(): Float

    data class Recognition(val id: String?,val title: String?,val confidence: Float?, var location: RectF?,var detectedClass: Int = 0) {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */


        /**
         * Display name for the recognition.
         */


        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */


        /**
         * Optional location within the source image for the location of the recognized object.
         */


        fun getLocation2(): RectF {
            return RectF(location)
        }

        fun setLocation2(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }
    }
}