import glob
import numpy as np

obs = []
acs = []


def vectorize_obs(obs):
    vect_obs = []
    for k,v in obs.items():
        for element in v:
            vect_obs.append(element)
    return vect_obs

for filepath in glob.iglob('good_tr/*.npz'):
	print(filepath)
	data = np.load(filepath)
	o = data['obs']
	a = data['acs']
    
	for i in range(o.shape[0]):
		#print(o[i])
		vect = vectorize_obs(o[i])
		obs.append(vect)
	for i in range(a.shape[0]):
		#print(a[i,:])
		ac = [x for x in a[i,:]]
		acs.append(ac)
   

#print('obs len', obs)
print('acs len', len(acs))
np.savez_compressed('human_expert_reach.npz', obs=obs, acs=acs)