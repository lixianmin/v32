package v32

import "testing"

/********************************************************************
created:    2023-10-13
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

func TestV32_Argmax(t *testing.T) {
	var v = (V32)([]float32{1, 2, 3, 4, 5, 6})

	if v.Argmax() != len(v)-1 {
		t.Fail()
	}
}
