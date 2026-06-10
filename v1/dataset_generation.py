from pathlib import Path

# r_values = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# k_values = [20, 30, 40, 50, 70, 90, 110, 130, 150, 170]
r_values = [0.1]
k_values = [20]


def fmt_num(x):
    if isinstance(x, int):
        return f"{x}"
    return f"{x:.1f}"


def case_block(case_id, r, kf, km):
    return f"""
! ===== CASE {case_id}: r={r:.1f}, Kf={kf}, Km={km} =====
/CLEAR,NOSTART
/FILNAME,case_{case_id:03d},1
/TITLE,Hexagonal cell case {case_id}
/UNITS,SI
PI=4*ATAN(1)
*AFUN,DEG
a=1
b=TAN(60)
r={r:.1f}
TT=100
TP=0
Kf={kf}
Km={km}

/PREP7
ANTYPE,STATIC
ET,1,PLANE77

MP,KXX,1,Kf
MP,KXX,2,Km

RECTNG,,a,,b
CYL4,0,0,r
CYL4,a,b,r

LSEL,S,,,1
LSEL,A,,,4
ASBL,2,ALL

LSEL,S,,,2
LSEL,A,,,3
ASBL,3,ALL

ASEL,S,,,5
ASEL,A,,,6
ADELE,ALL,,,1
ALLSEL,ALL

APTN,ALL

ASEL,S,,,3
AATT,2,,1,0
ALLSEL,ALL

ASEL,U,,,3
AATT,1,,1,0
ALLSEL,ALL

AESIZE,ALL,0.02
MSHAPE,0,2D
MSHKEY,0
AMESH,ALL

NSEL,S,LOC,Y,0
D,ALL,TEMP,TT

NSEL,S,LOC,Y,b
D,ALL,TEMP,0
ALLSEL,ALL

NSEL,S,LOC,X,0
F,ALL,HEAT,0

NSEL,S,LOC,X,a
F,ALL,HEAT,0
ALLSEL,ALL

/SOLU
SOLVE
FINISH

/POST1
SET,FIRST

ETABLE,,VOLU
ETABLE,,TF,X
ETABLE,,TF,Y
ETABLE,,TF,Z

SMULT,TFXV,VOLU,TFX
SMULT,TFYV,VOLU,TFY
SMULT,TFZV,VOLU,TFZ
SSUM

*GET,TOTVOL,SSUM,,ITEM,VOLU
*GET,TOTTFX,SSUM,,ITEM,TFXV
*GET,TOTTFY,SSUM,,ITEM,TFYV
*GET,TOTTFZ,SSUM,,ITEM,TFZV

TFX0=TOTTFX/TOTVOL
TFY0=TOTTFY/TOTVOL
TFZ0=TOTTFZ/TOTVOL

Ky=TFY0/(TT/b)
Kz=(Kf*2*PI*r*r/4 + Km*(a*b-2*PI*r*r/4))/(a*b)
PSI=2*PI*r*r/4/(a*b)

*GET,NNODE,NODE,0,COUNT
*GET,NELEM,ELEM,0,COUNT

*GET,NNODE,NODE,0,COUNT
*GET,NELEM,ELEM,0,COUNT

*CFOPEN,results_gex,txt,,APPEND
*VWRITE,r,Kf,Km,PSI,TFY0,Ky,Kz,NNODE,NELEM,TOTVOL
(F10.5,',',F10.5,',',F10.5,',',F12.6,',',E16.8,',',F12.6,',',F12.6,',',F12.0,',',F12.0,',',E16.8)
*CFCLOS
FINISH
"""


parts = []
parts.append("! Auto-generated APDL batch for 144 variants\n")
parts.append("/CWD,'C:\\temp'\n")
parts.append("*CFOPEN,results,csv\n")
parts.append("*VWRITE,'r','Kf','Km','Ky','Kz','PSI'\n")
parts.append("(A8,',',A8,',',A8,',',A14,',',A14,',',A14)\n")
parts.append("*CFCLOSE\n")

case_id = 1
for r in r_values:
    for kf in k_values:
        for km in k_values:
            parts.append(case_block(case_id, r, kf, km))
            case_id += 1

parts.append("\n/EXIT,NOSAVE\n")

content = "".join(parts)
path = Path(__file__).parent / "Gexagonal_144_variants.txt"
path.write_text(content, encoding="utf-8")
print(str(path))
