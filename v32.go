package v32

import (
	"github.com/chewxy/math32"
	"gorgonia.org/vecf32"
	"slices"
)

/********************************************************************
created:    2023-10-13
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

func Argmax(a []float32) int {
	return vecf32.Argmax(a)
}

func Argmin(a []float32) int {
	return vecf32.Argmin(a)
}

func Exp(a []float32) {
	for i := range a {
		a[i] = math32.Exp(a[i])
	}
}

func Log(a []float32) {
	for i := range a {
		a[i] = math32.Log(a[i])
	}
}

func Pow(a []float32, y float32) {
	for i := range a {
		a[i] = math32.Pow(a[i], y)
	}
}

func Scale(a []float32, s float32) {
	vecf32.Scale(a, s)
}

func Softmax(a []float32) {
	var maxVal = slices.Max(a)
	var expSum = float32(0.0)
	for i := range a {
		a[i] = math32.Exp(a[i] - maxVal)
		expSum += a[i]
	}

	vecf32.Scale(a, 1.0/expSum)
}
