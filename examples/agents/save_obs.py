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


   
def save():
	#print('obs len', obs)
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
			print('ac', ac)
			acs.append(ac)
		print('acs len', len(acs))
	np.savez_compressed('human_expert_reach.npz', obs=obs, acs=acs)

def load_and_concat(human, art):
	obs = []
	acs = []

	hd = np.load(human)
	ard = np.load(art)
	h_o = hd['obs']
	h_a = hd['acs']


	art_obs = ard['obs']
	art_acs = ard['acs']

	art_obs_vect = []
	for ep in art_obs:
		_ep  = []
		for o in ep[:,:]: 
			#o_v = vectorize_obs(o)
			art_obs_vect.append(o)

	acts_v = []
	for ep in art_acs:
		#print(ep.shape)
		_ep  = []
		for o in ep[:,:]: 
			#print(o)
			#o_v = vectorize_obs(o)
			acts_v.append(o)
	

	_obs = np.array(art_obs_vect)
	_acs = np.array(acts_v)

	obs = np.concatenate([h_o, _obs])
	acs = np.concatenate([h_a, _acs])
	print(obs.shape)
	print(acs.shape)
   # train_obs = obs_reshaped[int(test_data_length):,:]
   # test_obs = obs_reshaped[0:int(test_data_length),:]
	np.savez_compressed('human_gym_expert_mix_reach_2000.npz', obs=obs, acs=acs)




load_and_concat('human_expert_reach.npz', 'data_fetch_reach_random_2000.npz')