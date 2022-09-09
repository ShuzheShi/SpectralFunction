(* ::Package:: *)

(* ::Section:: *)
(*Head*)


BeginPackage["Figures`Color`"];
Green2::usage = ""
Blue2::usage = ""
Purple2::usage = ""
Gold2::usage = ""
Gold::usage = ""


(* ::Subsubsection:: *)
(*Options*)


Begin["`Private`"];


(* ::Section:: *)
(*Default Settings*)


Unprotect[Green];
Green=RGBColor[0,0.5,0];
Protect[Green];
Green2=RGBColor[0.513417, 0.72992, 0.440682];

Unprotect[Red];
Red=RGBColor[0.857359, 0.131106, 0.132128];
Protect[Red];
Unprotect[Blue];
Blue2=RGBColor[0.24487776, 0.34493928, 0.81045216];
Blue=RGBColor[0.1225, 0.1725, 0.9104];
Protect[Blue];

Unprotect[Orange];
Orange=RGBColor[1, 0.448541, 0];
Protect[Orange];

Unprotect[Purple];
Purple2=RGBColor[0.471412, 0.108766, 0.527016];
Protect[Purple];

Gold=RGBColor[1,0.669906,0];
Gold2=RGBColor[1,0.843137,0];


(* ::Section:: *)
(*End*)


End[ ]; (* End `Private` Context. *)


EndPackage[];
