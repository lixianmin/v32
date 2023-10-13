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

type V32 []float32

func (my V32) Argmax() int {
	return vecf32.Argmax(my)
}

func (my V32) Argmin() int {
	return vecf32.Argmin(my)
}

func (my V32) Clone() V32 {
	return slices.Clone(my)
}

func (my V32) Exp() {
	for i := range my {
		my[i] = math32.Exp(my[i])
	}
}

func (my V32) Log() {
	for i := range my {
		my[i] = math32.Log(my[i])
	}
}

func (my V32) Pow(p float32) {
	for i := range my {
		my[i] = math32.Pow(my[i], p)
	}
}

func (my V32) Scale(s float32) {
	vecf32.Scale(my, s)
}

func (my V32) Sum() float32 {
	return vecf32.Sum(my)
}

func (my V32) SoftMax() {
	var maxVal = slices.Max(my)
	var expSum = float32(0.0)
	for i := range my {
		my[i] = math32.Exp(my[i] - maxVal)
		expSum += my[i]
	}

	vecf32.Scale(my, 1.0/expSum)
}
