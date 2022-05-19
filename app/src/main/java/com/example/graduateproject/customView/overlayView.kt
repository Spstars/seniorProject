package com.example.graduateproject.customView

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import java.util.*

class overlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    private val callbacks: MutableList<DrawCallback> = LinkedList<DrawCallback>()


    fun addCallback(callback: DrawCallback) {
        callbacks.add(callback)
    }

    @Synchronized
    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        for (callback in callbacks) {
            callback.drawCallback(canvas)
        }
    }

    fun addCallback(callback: (Canvas) -> Unit) {

    }

    /** Interface defining the callback for client classes.  */
    interface DrawCallback {
        fun drawCallback(canvas: Canvas?)
    }
}