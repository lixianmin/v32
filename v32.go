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

func (my V32) Reduce(f func(a, b float32) float32, initial float32) (retVal float32) {
	return vecf32.Reduce(f, initial, my...)
}

func (my V32) Scale(s float32) {
	for i, v := range my {
		my[i] = v * s
	}
}

func (my V32) Sum() float32 {
	var sum = float32(0)
	for _, v := range my {
		sum += v
	}

	return sum
}

// SoftMax 激活函数: 将各个输出节点的输出值范围映射到[0, 1]，并且约束各个输出节点的输出值的和为1
func (my V32) SoftMax() {
	// 通常max()的理念是: 返回唯一的最大值, 这意味着模型的分类结果是确定且唯一的, 这AI分类中显然不合理.
	// SoftMax()的理念是: 不再确定唯一一个最大值，而是为每个输出分类的结果都赋予一个概率，表示属于每个类别的可能性。
	// 参考: https://zhuanlan.zhihu.com/p/105722023

	// 相比于probs[]/sum, 引入指数会将probs[]中的数值拉开极大差距 (在保持大小顺序关系的基础上)

	var maxValue = slices.Max(my)
	var expSum = float32(0.0)
	for i := range my {
		// 减去maxValue是为了规避数值溢出问题 (在probs[]中maxValue值远超其它的时候), 同时可保持大小顺序关系
		my[i] = math32.Exp(my[i] - maxValue)
		expSum += my[i]
	}

	my.Scale(1.0 / expSum)
}
