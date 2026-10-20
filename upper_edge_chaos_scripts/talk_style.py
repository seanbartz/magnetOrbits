"""Shared matplotlib style for the talk figures: large text for slides and LaTeX-style
math (matplotlib mathtext) so labels like $K_{\rm rot}$ show real subscripts."""
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size':20,'axes.labelsize':22,'axes.titlesize':22,'xtick.labelsize':19,'ytick.labelsize':19,
    'legend.fontsize':19,'legend.title_fontsize':19,'mathtext.fontset':'dejavusans',
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#666','axes.linewidth':1.2,
    'xtick.major.width':1.2,'ytick.major.width':1.2,'xtick.major.size':6,'ytick.major.size':6,
    'pdf.fonttype':42,'savefig.bbox':'tight'})
