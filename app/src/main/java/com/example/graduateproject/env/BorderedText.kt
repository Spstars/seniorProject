package com.example.graduateproject.env

import android.graphics.*
import android.graphics.Paint.Align
import java.util.*

public class BorderedText(var interiorPaint: Paint? = null ,var exteriorPaint: Paint? = null,var textSize: Float = 0f) {

    /**
     * Creates a left-aligned bordered text object with a white interior, and a black exterior with
     * the specified text size.
     *
     * @param textSize text size in pixels
     */

    constructor (textSize: Float) : this() {
        BorderedText(Color.WHITE, Color.BLACK, textSize)
    }


    /**
     * Create a bordered text object with the specified interior and exterior colors, text size and
     * alignment.
     *
     * @param interiorColor the interior text color
     * @param exteriorColor the exterior text color
     * @param textSize text size in pixels
     */


    constructor (interiorColor: Int, exteriorColor: Int, textSize: Float) : this() {
        interiorPaint = Paint()
        interiorPaint!!.setTextSize(textSize)
        interiorPaint!!.setColor(interiorColor)
        interiorPaint!!.setStyle(Paint.Style.FILL)
        interiorPaint!!.setAntiAlias(false)
        interiorPaint!!.setAlpha(255)
        exteriorPaint = Paint()
        exteriorPaint!!.setTextSize(textSize)
        exteriorPaint!!.setColor(exteriorColor)
        exteriorPaint!!.setStyle(Paint.Style.FILL_AND_STROKE)
        exteriorPaint!!.setStrokeWidth(textSize / 8)
        exteriorPaint!!.setAntiAlias(false)
        exteriorPaint!!.setAlpha(255)
        this.textSize = textSize
    }

    fun setTypeface(typeface: Typeface?) {
        interiorPaint!!.typeface = typeface
        exteriorPaint!!.typeface = typeface
    }

    fun drawText(canvas: Canvas, posX: Float, posY: Float, text: String?) {
        canvas.drawText(text!!, posX, posY, exteriorPaint!!)
        canvas.drawText(text, posX, posY, interiorPaint!!)
    }

    fun drawText(
        canvas: Canvas, posX: Float, posY: Float, text: String?, bgPaint: Paint?
    ) {
        val width = exteriorPaint!!.measureText(text)
        val textSize = exteriorPaint!!.textSize
        val paint = Paint(bgPaint)
        paint.style = Paint.Style.FILL
        paint.alpha = 160
        canvas.drawRect(posX, posY + textSize.toInt(), posX + width.toInt(), posY, paint)
        canvas.drawText(text!!, posX, posY + textSize, interiorPaint!!)
    }

    fun drawLines(canvas: Canvas, posX: Float, posY: Float, lines: Vector<String?>) {
        var lineNum = 0
        for (line in lines) {
            drawText(canvas, posX, posY - textSize * (lines.size - lineNum - 1), line)
            ++lineNum
        }
    }

    fun setInteriorColor(color: Int) {
        interiorPaint!!.color = color
    }

    fun setExteriorColor(color: Int) {
        exteriorPaint!!.color = color
    }

    fun setAlpha(alpha: Int) {
        interiorPaint!!.alpha = alpha
        exteriorPaint!!.alpha = alpha
    }

    fun getTextBounds(
        line: String?, index: Int, count: Int, lineBounds: Rect?
    ) {
        interiorPaint!!.getTextBounds(line, index, count, lineBounds)
    }

    fun setTextAlign(align: Align?) {
        interiorPaint!!.textAlign = align
        exteriorPaint!!.textAlign = align
    }
}