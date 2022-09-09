(* ::Package:: *)

(* ::Section:: *)
(*Head*)


BeginPackage["Figures`Basic`"];


SetAspectRatio::Usage = "SetAspectRatio[Plot/ListPlot/..., Ratio]"
SetImageSize::Usage = "SetImageSize[Plot/ListPlot/..., Ratio]"
SetGeneral::Usage = "SetGeneral[Plot/ListPlot/...]"


Begin["`Private`"];


(* ::Section:: *)
(*Common Setting*)


padding = Scaled[0.00];


SetGeneral[Target_]:=SetOptions[Target,
	Axes->False,
	AxesStyle->Directive[Black,AbsoluteThickness[1],AbsoluteDashing[{5,10}]],
	Frame->True,
	
	(*FrameTicksStyle->Directive[Black,20],
	FrameStyle->Directive[Black,25,AbsoluteThickness[1]],
	LabelStyle->Directive[Black],*)
	
	FrameTicksStyle->Directive[Black,20],
	FrameStyle->Directive[Black,25,AbsoluteThickness[1]],
	LabelStyle->Directive[Black,FontFamily->"Times New Roman"],
	
	AspectRatio->1,
	ImageMargins->{{0,0},{0,0}},
	ImagePadding->{{75,15},{75,15}},
	ImageSize->450,
	PlotRangePadding->padding
]


SetColor[Target_]:=SetOptions[Target,
	ColorFunction->"TemperatureMap"
]


SetImageSize[Target_,Ratio_:0.75]:=SetOptions[Target,
	Axes->False,
	Frame->True,
	FrameStyle->Directive[Black,AbsoluteThickness[1],28*Ratio],
	FrameTicksStyle->Directive[Black,22*Ratio],
	AspectRatio->1,
	ImageMargins->{{0,0},{0,0}},
	ImagePadding->{{75,15},{75,15}}*Ratio,
	ImageSize->450*Ratio,
	PlotRangePadding->padding
]


SetAspectRatio[Target_,Ratio_:0.8]:=SetOptions[Target,
	Axes->False,
	Frame->True,
	FrameStyle->Directive[Black,28,AbsoluteThickness[1]],
	FrameTicksStyle->Directive[Black,22],
	AspectRatio->Ratio,
	ImageMargins->{{0,0},{0,0}},
	ImagePadding->{{75,15},{75,15}Ratio},
	ImageSize->450/Ratio,
	PlotRangePadding->padding
]


(* ::Section:: *)
(*Set General (Geometry)*)


SetGeneral[Plot];
SetGeneral[LogPlot];
SetGeneral[LogLogPlot];
SetGeneral[LogLinearPlot];

SetGeneral[ListPlot];
SetGeneral[ListLogPlot];
SetGeneral[ListLogLogPlot];
SetGeneral[ListLogLinearPlot];

SetGeneral[ListLinePlot];

SetGeneral[ContourPlot];
SetGeneral[DensityPlot];
SetGeneral[VectorPlot];
SetGeneral[StreamPlot];
SetGeneral[StreamDensityPlot];

SetGeneral[ListContourPlot];
SetGeneral[ListDensityPlot];
SetGeneral[ListVectorPlot];

SetGeneral[ListStreamPlot];
SetGeneral[ListStreamDensityPlot];


(* ::Section:: *)
(*Set Defaulted Color Function*)


SetColor[ContourPlot];
SetColor[DensityPlot];

SetColor[ListContourPlot];
SetColor[ListDensityPlot];

SetOptions[Style,FontFamily->"Times New Roman",FontSize->25]


(* ::Section:: *)
(*End*)


End[ ]; (* End `Private` Context. *)


EndPackage[];
