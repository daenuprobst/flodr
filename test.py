from mnist1d.data import get_dataset_args, make_dataset
args = get_dataset_args()
args.train_split = 1
args.num_samples = 1000
d = make_dataset(args)
X,labels = d['x'], d['y']
print("training model")
# print(x.shape, y.shape)

import numpy as np
from flodr import FloDR


model = FloDR(w=2.0, random_state=0, density=True)
Y = model.fit_transform(X)                 # (n, 2)
Z = model._coords
Y_new = model.transform(X[:10])            # new points in one forward pass
X_back = model.inverse_transform(Y[:10])   # decode screen positions to input space
X_lat = model.inverse_coords(Z)            # decode full latent coordinates
print("calculating logp")
logp = model.score_samples(X)              # exact log density of the flow

print("diagnostics")
sigma = model.conditional_spread()
print(1)         # per-point hidden spread, input units
atyp = model.atypicality()                 # how unusual each point is off the layout
print(2)
cal = model.spread_calibration()           # calibration certificate for the spread
print(3)
field, cert = model.hidden_contrast(labels)  # what labels the layout hides
print(4)
report = model.diagnostics(G=labels)       # both fields with their certificates