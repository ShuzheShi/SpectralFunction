(* ::Package:: *)

(* ::Section:: *)
(*Head*)


BeginPackage["Figures`Ticks`"];


(* ::Subsubsection:: *)
(*Functions*)


LogTicks::Usage = "LogTicks[min,max,(show # or not),Options], :{\"SubMajorTicks\"->True/False}]"
LinearTicks::Usage = "LinearTicks[min,max,(show # or not)]"


(* ::Subsubsection:: *)
(*Options*)


Begin["`Private`"];


(* ::Section:: *)
(*LogTicks*)


Options[LogTicks]={TicksColor->Black,TicksOrientation->0,MaxLogScale->3,Division->5,
	MajorTicksLength->0.04,SubMajorTicksLength->0.025,MinorTicksLength->0.01};
LogTicks[min_?NumericQ,max_?NumericQ,ShowQ_:True,opts:OptionsPattern[]]:=
Module[{LogTicksList,LogMajorTicksList,LogSubMajorTicksList,LogMinorTicksList,nstep,nstep2,
	tickscolor,majorTicksLength,minorTicksLength,subMajorTicksLength,ticksOrientation,maxLogScale,divide,
	LogMajorTicks,LogSubMajorTicks,LogMinorTicks,getKey,Scale1,Scale2},
	
	tickscolor=OptionValue[TicksColor];
	majorTicksLength=OptionValue[MajorTicksLength];
	minorTicksLength=OptionValue[MinorTicksLength];
	subMajorTicksLength=OptionValue[SubMajorTicksLength];
	ticksOrientation=OptionValue[TicksOrientation];
	maxLogScale=OptionValue[MaxLogScale];
	divide=OptionValue[Division];
	
	(*tickscolor = Black;*)
	If[Log10[max/min]<=maxLogScale,(* {1,2,5,10,20,..} *)
		LogTicksList=Flatten[Table[i*10^j,{j,Floor[Log10[min]],Ceiling[Log10[max]]-1},{i,9}]];
		LogMajorTicksList=Table[10^j,{j,Floor[Log10[min]],Ceiling[Log10[max]]-1}];
		LogSubMajorTicksList=Select[Complement[LogTicksList,LogMajorTicksList],IntegerQ[Log10[2*#]]||IntegerQ[Log10[5*#]]&];
		LogMinorTicksList=Complement[LogTicksList,LogMajorTicksList,LogSubMajorTicksList];
		LogSubMajorTicks=Table[{i,If[ShowQ,If[IntegerQ[i],ToString[i],N[i]],""],
			{subMajorTicksLength*(1-ticksOrientation),subMajorTicksLength*ticksOrientation},tickscolor},{i,LogSubMajorTicksList}];
		LogMajorTicks=Table[{i,If[ShowQ,If[IntegerQ[i],ToString[i],N[i]],""],
			{majorTicksLength*(1-ticksOrientation),majorTicksLength*ticksOrientation},tickscolor},{i,LogMajorTicksList}];
	,If[Log10[max/min]<=10,(*10^1, 10^2, ...*)
		LogTicksList=Flatten[Table[i*10^j,{j,Floor[Log10[min]],Ceiling[Log10[max]]-1},{i,9}]];
		LogMajorTicksList=Table[10^j,{j,Floor[Log10[min]],Ceiling[Log10[max]]-1}];
		LogSubMajorTicksList={};
		LogMinorTicksList=Complement[LogTicksList,LogMajorTicksList];
		LogMajorTicks=Table[{i,If[ShowQ,Superscript["10",ToString[Log10[i]]],""],
			{majorTicksLength*(1-ticksOrientation),majorTicksLength*ticksOrientation},tickscolor},{i,LogMajorTicksList}];
		LogSubMajorTicks=Table[{i,"",
			{subMajorTicksLength*(1-ticksOrientation),subMajorTicksLength*ticksOrientation},tickscolor},{i,LogSubMajorTicksList}];
	,(*10^3n,...*)
		nstep = Ceiling[Log10[max/min]/divide,3];
		nstep2 = 3*Nearest[Divisors[nstep/3],nstep/12][[1]];
		LogMajorTicksList=Table[10^(j*nstep),{j,Floor[Log10[min]/nstep],Ceiling[Log10[max]/nstep]-1}];
		LogTicksList=Table[10^(j*nstep2),{j,Floor[Log10[min]/nstep2],Ceiling[Log10[max]/nstep2]-1}];
		LogMajorTicks=Table[{i,If[ShowQ,Superscript["10",ToString[Log10[i]]],""],
			{majorTicksLength*(1-ticksOrientation),majorTicksLength*ticksOrientation},tickscolor},{i,LogMajorTicksList}];
		LogMinorTicksList=Complement[LogTicksList,LogMajorTicksList];
		LogSubMajorTicksList={};
		LogSubMajorTicks=Table[{i,"",
			{subMajorTicksLength*(1-ticksOrientation),subMajorTicksLength*ticksOrientation},tickscolor},{i,LogSubMajorTicksList}];
	];];
	LogMinorTicks=Table[{i,"",{minorTicksLength*(1-ticksOrientation),minorTicksLength*ticksOrientation},tickscolor},{i,LogMinorTicksList}];
	Return[Join[LogMajorTicks,LogSubMajorTicks,LogMinorTicks]];
]


(* ::Section:: *)
(*LinearTicks*)


Options[LinearTicks]={TicksColor->Black,Division->5,MajorTicksLength->0.04,MinorTicksLength->0.01,TicksOrientation->0};
LinearTicks[min_?NumericQ,max_?NumericQ,ShowQ_:True,opts:OptionsPattern[]]:=
Module[{Dx,dx,LabelDigits,TicksList,MajorTicksList,MinorTicksList,
	tickscolor,divide,majorTicksLength,minorTicksLength,ticksOrientation,
	MajorTicks,MinorTicks},
	tickscolor=OptionValue[TicksColor];
	divide=OptionValue[Division];
	majorTicksLength=OptionValue[MajorTicksLength];
	minorTicksLength=OptionValue[MinorTicksLength];
	ticksOrientation=OptionValue[TicksOrientation];
	
	Dx=10^(Floor[#]+Nearest[{Log10[1],Log10[2],Log10[5],Log10[10]},#-Floor[#]][[1]])&[Log10[(max-min)/divide]];
	dx=10^(Floor[#]+Nearest[{Log10[1],Log10[2],Log10[5],Log10[10]},#-Floor[#]][[1]])&[Log10[Dx/5]];
	LabelDigits=Floor[Log10[Dx]];
	TicksList=Table[i,{i,Floor[min,dx],Ceiling[max,dx],dx}];
	MajorTicksList=Select[TicksList,IntegerQ[#/Dx]&];
	MinorTicksList=Complement[TicksList,MajorTicksList];
	MajorTicks=Table[{i,If[!ShowQ,"",If[LabelDigits>=0,ToString[i],ToString[NumberForm[N[i],{6,-LabelDigits}]]]],
		{majorTicksLength*(1-ticksOrientation),majorTicksLength*ticksOrientation},tickscolor},{i,MajorTicksList}];
	MinorTicks=Table[{i,"",
		{minorTicksLength*(1-ticksOrientation),minorTicksLength*ticksOrientation},tickscolor},{i,MinorTicksList}];
	Return[Join[MajorTicks,MinorTicks]];
];


(* ::Section::Closed:: *)
(*SetOptions*)


SetOptions[ListLinePlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[Plot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[DensityPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListDensityPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ContourPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListContourPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[StreamPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[StreamDensityPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListStreamPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListStreamDensityPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[VectorPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[ListVectorPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];


SetOptions[ListLogPlot,FrameTicks->{{LogTicks,LogTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];
SetOptions[LogPlot,FrameTicks->{{LogTicks[#1,#2,True]&,LogTicks[#1,#2,False]&},{LinearTicks,LinearTicks[#1,#2,False]&}}];


SetOptions[ListLogLinearPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LogTicks,LogTicks[#1,#2,False]&}}];
SetOptions[LogLinearPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LogTicks,LogTicks[#1,#2,False]&}}];
(*SetOptions[LogLinearPlot,FrameTicks->{{LinearTicks,LinearTicks[#1,#2,False]&},{LogTicks[Exp[#1],Exp[#2],True]&,LogTicks[Exp[#1],Exp[#2],False]&}}];*)


SetOptions[ListLogLogPlot,FrameTicks->{{LogTicks[#1,#2,True]&,LogTicks[#1,#2,False]&},{LogTicks[#1,#2,True]&,LogTicks[#1,#2,False]&}}];
SetOptions[LogLogPlot,FrameTicks->{{LogTicks[#1,#2,True]&,LogTicks[#1,#2,False]&},{LogTicks[#1,#2,True]&,LogTicks[#1,#2,False]&}}];
(*SetOptions[LogLogPlot,FrameTicks->{{LogTicks[Exp[#1],Exp[#2],True]&,LogTicks[Exp[#1],Exp[#2],False]&},{LogTicks[Exp[#1],Exp[#2],True]&,LogTicks[Exp[#1],Exp[#2],False]&}}];*)


(* ::Section:: *)
(*End*)


End[ ]; (* End `Private` Context. *)
SetAttributes[
	{TicksColor,Division,TicksOrientation,MaxLogScale,
	MajorTicksLength,SubMajorTicksLength,MinorTicksLength},
{Protected}];


EndPackage[];
