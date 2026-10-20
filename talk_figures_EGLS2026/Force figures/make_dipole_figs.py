import numpy as np, textwrap
d2r=np.radians
TH,P1,P2,SEP,R=15,35,80,5.4,0.85
rh=np.array([np.cos(d2r(TH)),np.sin(d2r(TH))])
m1=np.array([np.cos(d2r(P1)),np.sin(d2r(P1))]); m2=np.array([np.cos(d2r(P2)),np.sin(d2r(P2))])
X2=SEP*rh
B=3*(m1@rh)*rh-m1
cen=((m1@m2)-5*(m1@rh)*(m2@rh))*rh; c=(m1@rh)*m2; d=(m2@rh)*m1; F=cen+c+d
ang=lambda v: np.degrees(np.arctan2(v[1],v[0]))
f=lambda v: f"({v[0]:.4f},{v[1]:.4f})"

pre=r"""\documentclass[border=8pt]{standalone}
\usepackage{tikz,amsmath,amssymb,bm}
\usetikzlibrary{arrows.meta,calc,angles,quotes,decorations.markings}
\begin{document}
\begin{tikzpicture}[>={Stealth[length=7pt,width=5pt]},
  mom/.style={->,line width=1.8pt}, vec/.style={->,line width=1.1pt},
  aux/.style={densely dashed,gray!70},
  lbl/.style={fill=white,fill opacity=0.85,text opacity=1,inner sep=1pt,font=\footnotesize}]
"""
post="\\end{tikzpicture}\n\\end{document}\n"
def spheres(op=1):
    return (f"\\coordinate (P1) at (0,0);\\coordinate (P2) at {f(X2)};\n"
     f"\\shade[ball color=red!65!white,opacity={op}] (P1) circle ({R});\n"
     f"\\shade[ball color=blue!45!white,opacity={op}] (P2) circle ({R});\n")
def moments():
    return (f"\\draw[mom,red!60!black] (P1) ++({P1+180}:1.25) -- ++({P1}:2.5) node[above=1pt] {{$\\bm\\mu_1$}};\n"
            f"\\draw[mom,blue!60!black] (P2) ++({P2+180}:1.25) -- ++({P2}:2.5) node[above=1pt] {{$\\bm\\mu_2$}};\n")

# ---------- Fig 1: geometry
s=pre+r"\coordinate (O) at (-0.6,-2.6);"+"\n"+spheres()+rf"""
\draw[vec,gray!80!black] (O) -- (P1) node[midway,left=1pt] {{$\mathbf r_1$}};
\draw[vec,gray!80!black] (O) -- (P2) node[pos=0.6,below right=-1pt] {{$\mathbf r_2$}};
\fill (O) circle (1.2pt) node[below left] {{$O$}};
\draw[aux] (P1) -- ++(1.8,0) coordinate (X1);
\draw[aux] (P2) -- ++(1.6,0) coordinate (X2);
\draw[vec] ($(P1)!{R} cm!(P2)$) -- ($(P2)!{R} cm!(P1)$) node[pos=0.62,below=3pt,sloped] {{$\mathbf r=\mathbf r_2-\mathbf r_1$}};
\draw[vec,line width=1.3pt] ($(P1)!0.42!(P2)+({TH+90}:0.32)$) -- ++({TH}:0.9) node[above=1pt] {{$\hat{{\mathbf r}}=\mathbf r/r$}};
"""+moments()+rf"""
\coordinate (T1) at ({TH}:1.6);
\pic[draw,angle radius=1.6cm,"$\theta$",angle eccentricity=1.13] {{angle=X1--P1--T1}};
\coordinate (A1) at ({P1}:1.0);
\pic[draw,red!60!black,angle radius=0.5cm,"$\phi_1$"lbl,angle eccentricity=2.0] {{angle=X1--P1--A1}};
\coordinate (A2) at ($(P2)+({P2}:1.0)$);
\pic[draw,blue!60!black,angle radius=0.5cm,"$\phi_2$"lbl,angle eccentricity=1.9] {{angle=X2--P2--A2}};
"""+post
open('fig1_geometry.tex','w').write(s)

# ---------- Fig 2: field of dipole 1 and the energy
alpha=np.arccos(np.clip(m1@rh,-1,1)); C2=SEP/np.sin(alpha)**2
lines=[]
def fieldline(C,side):
    a0=np.arcsin(np.sqrt(R/C)); al=np.linspace(a0,np.pi-a0,400)
    r=C*np.sin(al)**2
    xl=r*np.cos(al); yl=side*r*np.sin(al)
    e1=m1; e2=np.array([-m1[1],m1[0]])
    pts=[xl[i]*e1+yl[i]*e2 for i in range(len(al))]
    return " -- ".join(f(p) for p in pts[::4]+[pts[-1]])
fl=""
for C in [1.5,2.4,3.8,6.0,10.0,18.0]:
    for sd in [1,-1]:
        fl+=f"\\draw[orange!70!black,opacity=0.45,line width=0.6pt,postaction={{decorate}}] {fieldline(C,sd)};\n"
side=np.sign(np.cross(m1,rh))
fl+=f"\\draw[orange!80!black,line width=1.0pt,postaction={{decorate}}] {fieldline(C2,side)};\n"
s=pre+r"""\tikzset{decoration={markings,mark=at position 0.5 with {\arrow{>}}}}
\begin{scope}\clip (-3.6,-3.4) rectangle (8.2,4.4);
"""+fl+r"\end{scope}"+"\n"+spheres()+moments()+rf"""
\draw[vec,orange!80!black,line width=1.6pt] (P2) -- ++({ang(B):.3f}:1.9) node[right] {{$\mathbf B_1$}};
\draw[aux] ($(P1)!{R} cm!(P2)$) -- ($(P2)!{R} cm!(P1)$);
\node[draw,rounded corners,fill=white,align=left,anchor=north west,font=\small] at (8.6,4.2) {{%
 $\displaystyle \mathbf B_1(\mathbf r_2)=\frac{{\mu_0}}{{4\pi r^3}}\Big[3(\bm\mu_1\!\cdot\hat{{\mathbf r}})\hat{{\mathbf r}}-\bm\mu_1\Big]$\\[12pt]
 $\displaystyle U_{{12}}=-\bm\mu_2\!\cdot\mathbf B_1(\mathbf r_2)$\\[6pt]
 $\displaystyle \hphantom{{U_{{12}}}}=-\frac{{\mu_0}}{{4\pi r^3}}\Big[3(\bm\mu_1\!\cdot\hat{{\mathbf r}})(\bm\mu_2\!\cdot\hat{{\mathbf r}})-\bm\mu_1\!\cdot\bm\mu_2\Big]$\\[10pt]
 \footnotesize Orange: field lines of $\bm\mu_1$ (outside a uniformly magnetized\\[-1pt]
 \footnotesize sphere the field is exactly that of a point dipole). The darker\\[-1pt]
 \footnotesize line passes through magnet 2. $U_{{12}}$ is lowest when $\bm\mu_2\parallel\mathbf B_1$.}};
"""+post
open('fig2_field_energy.tex','w').write(s)

# ---------- Fig 3: force terms
S=2.1  # cm per unit of 3 mu0 mu^2/(4 pi r^4)
q0=X2; q1=q0+S*cen; q2=q1+S*c; q3=q2+S*d
s=pre+spheres(0.35)+moments()+rf"""
\draw[aux] (P1) -- (P2);
\draw[vec,violet,line width=1.4pt] {f(q0)} -- {f(q1)} node[midway,below=4pt,lbl,text=violet] {{central}};
\draw[vec,blue!60!black,line width=1.4pt] {f(q1)} -- {f(q2)} node[midway,left=2pt,lbl,text=blue!60!black] {{$(\bm\mu_1\!\cdot\hat{{\mathbf r}})\bm\mu_2$}};
\draw[vec,red!60!black,line width=1.4pt] {f(q2)} -- {f(q3)} node[midway,above left=0pt,lbl,text=red!60!black] {{$(\bm\mu_2\!\cdot\hat{{\mathbf r}})\bm\mu_1$}};
\draw[vec,green!45!black,line width=1.9pt] {f(q0)} -- {f(q3)} node[pos=0.55,right=3pt,lbl,text=green!40!black] {{$\mathbf F_2$}};
\draw[vec,green!45!black,line width=1.9pt] (P1) -- ++{f(-S*F)} node[below right=-2pt] {{$\mathbf F_1=-\mathbf F_2$}};
\fill (P2) circle (1.3pt);
\node[draw,rounded corners,fill=white,align=left,anchor=north west,font=\small] at (7.2,4.0) {{%
 $\displaystyle \mathbf F_2=\nabla(\bm\mu_2\!\cdot\mathbf B_1)=-\nabla U_{{12}}$\\[10pt]
 $\displaystyle \mathbf F_2=\frac{{3\mu_0}}{{4\pi r^4}}\Big[\underbrace{{\big((\bm\mu_1\!\cdot\bm\mu_2)-5(\bm\mu_1\!\cdot\hat{{\mathbf r}})(\bm\mu_2\!\cdot\hat{{\mathbf r}})\big)\hat{{\mathbf r}}}}_{{\color{{violet}}\text{{central}}}}$\\[4pt]
 $\displaystyle \hphantom{{\mathbf F_2=\frac{{3\mu_0}}{{4\pi r^4}}\Big[}}+\underbrace{{{{\color{{blue!60!black}}(\bm\mu_1\!\cdot\hat{{\mathbf r}})\bm\mu_2}}+{{\color{{red!60!black}}(\bm\mu_2\!\cdot\hat{{\mathbf r}})\bm\mu_1}}}}_{{\text{{non-central}}}}\Big]$\\[10pt]
 \footnotesize Terms drawn tip to tail at magnet 2, to scale, for\\[-1pt]
 \footnotesize $\theta={TH}^\circ$, $\phi_1={P1}^\circ$, $\phi_2={P2}^\circ$. The non-central terms tilt\\[-1pt]
 \footnotesize $\mathbf F_2$ off the line of centers, so $\mathbf r\times\mathbf F_2\neq 0$: orbital\\[-1pt]
 \footnotesize and spin angular momentum are exchanged.}};
"""+post
open('fig3_force_terms.tex','w').write(s)

# ---------- Fig 4: special cases (r-hat along +x)
cases=[("Head-to-tail",0,0),("Side by side, parallel",90,90),("Side by side, antiparallel",90,-90),("T configuration",0,90)]
body=""; sep4=4.2; R4=0.55; S4=0.55
for k,(name,a1,a2) in enumerate(cases):
    ox=k*7.6
    u1=np.array([np.cos(d2r(a1)),np.sin(d2r(a1))]); u2=np.array([np.cos(d2r(a2)),np.sin(d2r(a2))]); e=np.array([1,0])
    FF=((u1@u2)-5*(u1@e)*(u2@e))*e+(u1@e)*u2+(u2@e)*u1
    U=-(3*(u1@e)*(u2@e)-u1@u2)
    p1=np.array([ox,0]); p2=np.array([ox+sep4,0])
    def n(x): x=round(x); return "0" if x==0 else (f"{x:+d}".replace("-","-"))
    def fv(v): return f"({n(v[0])},\\,{n(v[1])})"
    body+=(f"\\shade[ball color=red!65!white] {f(p1)} circle ({R4});\n"
      f"\\shade[ball color=blue!45!white] {f(p2)} circle ({R4});\n"
      f"\\draw[mom,red!60!black] {f(p1-0.8*u1)} -- ++{f(1.6*u1)};\n"
      f"\\draw[mom,blue!60!black] {f(p2-0.8*u2)} -- ++{f(1.6*u2)};\n"
      f"\\draw[vec,green!45!black,line width=1.6pt] {f(p2+0.9*FF/np.linalg.norm(FF))} -- ++{f(S4*FF)};\n"
      f"\\draw[vec,green!45!black,line width=1.6pt] {f(p1-0.9*FF/np.linalg.norm(FF))} -- ++{f(-S4*FF)};\n"
      f"\\node[font=\\small\\bfseries] at {f(np.array([ox+sep4/2,2.0]))} {{{name}}};\n"
      f"\\node[font=\\small,align=center] at {f(np.array([ox+sep4/2,-2.1]))} {{$\\mathbf F_2\\propto{fv(FF)}$\\\\[2pt] $U_{{12}}\\propto {n(U)}$}};\n")
s=pre+body+rf"""
\draw[vec] (-0.6,-3.2) -- ++(0.9,0) node[right] {{$\hat{{\mathbf r}}$}};
\node[font=\footnotesize,align=left,anchor=west] at (1.3,-3.2) {{%
 Forces in units of $3\mu_0\mu^2/4\pi r^4$, written as $(\hat{{\mathbf r}},\perp)$ components of $\mathbf F_2$; $U$ in units of $\mu_0\mu^2/4\pi r^3$.
 Green arrows: $\mathbf F_1$ (left) and $\mathbf F_2$ (right).\\
 Head-to-tail is the most attractive. In the T configuration the force is purely non-central: $\mathbf F_1$ and $\mathbf F_2$ form a couple.}};
"""+post
open('fig4_special_cases.tex','w').write(s)
print(F, cen, c, d)
