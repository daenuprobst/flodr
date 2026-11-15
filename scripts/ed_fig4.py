import fig2

# in eight columns a one-line "Parametric UMAP" runs into the next title
fig2.TITLE["PCUMAP"] = "Parametric\nUMAP"
methods = "DREAMS TriMap PaCMAP LocalMAP PHATE PyMDE PCUMAP SQuadMDS".split()
fig2.draw(methods, "ed_fig4.pdf")
