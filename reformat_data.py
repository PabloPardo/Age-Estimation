import numpy as np
from PIL import Image
from pandas.io.parsers import read_csv
from utils import outliers_filter

# -------------------------------------------------------------------------------
# These script reads the HuPBA AGE data in csv format (votes and pictures tables)
# and creates an specular image from all the images filtering by number of votes
# and creates a csv file with the information of the real age and apparent age.
# -------------------------------------------------------------------------------

# GLOBAL VARIABLES
path_fl = '../../Databases/Aging DB/AGE HuPBA/HuPBA_AGE_data.csv'
path_fl_votes = '../../Databases/Aging DB/AGE HuPBA/votes_data.csv'
path_new_fl = '../../Databases/Aging DB/AGE HuPBA/HuPBA_AGE_data_extended.csv'
path_db = '../../Databases/Aging DB/AGE HuPBA/original/'
path_db_ext = '../../Databases/Aging DB/AGE HuPBA/extended/'

min_num_votes = 5
margin_outliers = 10

# Read Data
with open(path_fl) as f:
    content = f.readlines()

# Read Votes
votes = read_csv(path_fl_votes)
votes_hist = {}
for i in set(votes['id_pic'].values):
    # Filter outliers
    votes_hist['%i' % i] = outliers_filter(votes.query('id_pic == %i' % i)['vote'].values, margin=margin_outliers)

# Cut off the pictures with less votes than 'min_num_votes'
votes_hist = {k: v for k, v in votes_hist.items() if len(v) >= min_num_votes}

real_age = []
apparent_age = []
with open(path_new_fl, 'w') as f:
    for i in range(1, len(content)):
        l = content[i]
        aux = l.split(',')

        if aux[0] in votes_hist:
            real_age.append(int(aux[3]))
            apparent_age.append(round(np.mean(votes_hist[aux[0]])))
            im = aux[2].decode('latin_1')

            # Rename the image in the csv and write it
            new_name = 'image_%s.png' % i
            aux[2] = new_name
            aux[4] = str(apparent_age[-1])
            new_l = ','.join(aux)
            f.write(new_l)

            # Add the specular image in the csv
            new_name_spec = 'image_%s_spec.png' % i
            aux[1] = new_name_spec
            new_l_spec = ','.join(aux)
            f.write(new_l_spec)

            # Convert the image into PNG if its not
            aux2 = im.split('.')
            if aux2[-1] != 'png':
                aux2[-1] = 'png'
                im = '.'.join(aux2)

            # Load Image and convert it to GrayScale and save it
            I = Image.open(path_db + im).convert('LA')
            I.save(path_db_ext + new_name)

            # Flip image and save it
            I_spec = I.transpose(Image.FLIP_LEFT_RIGHT)
            I_spec.save(path_db_ext + new_name_spec)

            print new_name

import matplotlib.pyplot as plt

hist, bins = np.histogram(real_age, bins=max(real_age))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title('Real Age Distribution')
plt.show()


hist, bins = np.histogram(apparent_age, bins=max(apparent_age))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title('Apparent Age Distribution')
plt.show()
