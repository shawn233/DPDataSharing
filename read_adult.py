import numpy as np

def gen_samples(path):
	data=np.genfromtxt(fname=path,delimiter=',',dtype=str,autostrip=True)
	dict={1:{'Private':0,'Self-emp-not-inc':1,'Self-emp-inc':2,'Federal-gov':3,'Local-gov':4,'State-gov':5,'Without-pay':6,'Never-worked':7,'?':-1},
	      3:{'Bachelors':0,'Some-college':0,'11th':0,'HS-grad':0,'Prof-school':0,'Assoc-acdm':0,'Assoc-voc':0,'9th':0,'7th-8th':0,'12th':0,'Masters':0,'1st-4th':0,'10th':0,'Private':0,'Doctorate':0,'5th-6th':0,'Preschool':0,'?':-1},
	      5:{'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3,'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6,'?':-1},6:{'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11,'Protective-serv':12,'Armed-Forces':13,'?':-1},7:{'Wife':0,'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5,'?':-1},8:{'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4,'?':-1},9:{'Female':0, 'Male':1,'?':-1},13:{'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':25, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40,'?':-1},14:{'>50K':0, '<=50K':1}}
	for i in range(np.shape(data)[0]):
	    for j in range(np.shape(data)[1]):
	        
	        if j in dict.keys():
	            data[i][j]=dict[j][data[i][j]]
	        else: data[i][j]=float(data[i][j])
	            
	#data=data.astype(np.float)

	return data[:,:-1],data[:,14]

