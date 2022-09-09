(* ::Package:: *)

(* ::Section:: *)
(*Head*)


BeginPackage["Figures`MultiPanel`"];

MultiPanel::usage = "MultiPanel[Figures,Options], \[LineSeparator]Creates a graphics grid sharing x and/or y axis.\[LineSeparator]Options include: Epilog, AspectRatio, PlotRange, FrameStyle, FrameLabel, FrameTicks, FrameTicksStyle."


Begin["`Private`"];


(* ::Section:: *)
(*MultiPanel*)


Options[MultiPanel]=Join[{ImageSize->400,ImagePadding->{{80,20},{80,20}},AspectRatio->0.5,
xRatios->1,yRatios->1,PlotRange->All},Options[ListPlot]];
MultiPanel[figs_,opts:OptionsPattern[]]:=
Module[
	{dim,dimensions,Figs,
	imagesize,lpad,rpad,bpad,tpad,aspectratio,
	Imagesize,Lpad,Rpad,Tpad,Bpad,
	IMAGESIZE,ASPECTRATIO,ImageX,ImageY,Fig,XRatios,YRatios},
	dim=Dimensions[figs];
	dimensions=If[Length[dim]==1,{dim[[1]],1},dim];
	Figs=If[Length[dim]==1,Table[{figs[[i1]]},{i1,dimensions[[1]]}],figs];
	imagesize=OptionValue[ImageSize];
	{{lpad,rpad},{bpad,tpad}}=OptionValue[ImagePadding];
	aspectratio=OptionValue[AspectRatio];
	XRatios=OptionValue[xRatios];
	YRatios=OptionValue[yRatios];
	If[!VectorQ[XRatios],XRatios=Table[1,{i,dimensions[[2]]}]];
	If[!VectorQ[YRatios],YRatios=Table[1,{i,dimensions[[1]]}]];
	If[Length[XRatios]!=dimensions[[2]],XRatios=Table[1,{i,dimensions[[2]]}]];
	If[Length[YRatios]!=dimensions[[1]],YRatios=Table[1,{i,dimensions[[1]]}]];
	XRatios=dimensions[[2]]*XRatios/Total[XRatios];
	YRatios=dimensions[[1]]*YRatios/Total[YRatios];
	Lpad=Table[0,{i1,dimensions[[1]]},{i2,dimensions[[2]]}];
	Rpad=Table[0,{i1,dimensions[[1]]},{i2,dimensions[[2]]}];
	Tpad=Table[0,{i1,dimensions[[1]]},{i2,dimensions[[2]]}];
	Bpad=Table[0,{i1,dimensions[[1]]},{i2,dimensions[[2]]}];
	Lpad[[;;,1]]=Table[lpad,{i1,dimensions[[1]]}];
	Rpad[[;;,-1]]=Table[rpad,{i1,dimensions[[1]]}];
	Tpad[[1]]=Table[tpad,{i1,dimensions[[2]]}];
	Bpad[[-1]]=Table[bpad,{i1,dimensions[[2]]}];
	Imagesize=Lpad+Rpad+Table[imagesize*XRatios[[i2]],{i1,dimensions[[1]]},{i2,dimensions[[2]]}];

	IMAGESIZE=dimensions[[2]]imagesize+rpad+lpad;
	ASPECTRATIO=(dimensions[[1]]imagesize*aspectratio+tpad+bpad)/(dimensions[[2]]imagesize+rpad+lpad);
	ImageX=(Accumulate[XRatios]-0.5*XRatios)*imagesize;
	ImageY=(dimensions[[1]]-Accumulate[YRatios]+0.5*YRatios)*imagesize*aspectratio;

	Fig=Table[
		Show[Figs[[i1,i2]],
		ImageSize->Imagesize[[i1,i2]],
		ImageMargins->{{0,0},{0,0}},
		AspectRatio->aspectratio*YRatios[[i1]]/XRatios[[i2]],
		ImagePadding->{{Lpad[[i1,i2]],Rpad[[i1,i2]]},{Bpad[[i1,i2]],Tpad[[i1,i2]]}},
		Evaluate[FilterRules[{opts},{PlotRange,FrameStyle,FrameLabel,FrameTicks,FrameTicksStyle}]]
	],
	{i1,dimensions[[1]]},{i2,dimensions[[2]]}];

	Graphics[{},PlotRange->{{-lpad,dimensions[[2]]imagesize+rpad},{-bpad,dimensions[[1]]imagesize aspectratio+tpad}},
		Epilog->Join[OptionValue[Epilog],
		Table[Inset[Fig[[i1,i2]],{ImageX[[i2]],ImageY[[i1]]},Scaled[{0.5,0.5}]],{i1,dimensions[[1]]},{i2,dimensions[[2]]}]],
		ImageSize->IMAGESIZE,
		AspectRatio->ASPECTRATIO,
		Frame->False,
		ImagePadding->0,
		ImageMargins->0
	]
]


(* ::Section:: *)
(*End*)


End[ ]; (* End `Private` Context. *)
SetAttributes[
	{xRatios,yRatios}
,{Protected}]


EndPackage[];
