"""
Initialement méthode qui identifiait les objets de taille supérieur à 0.5% afin de les supprimer de la surface de nuage.
Mais la méthode contour englobe
"""


import cv2
import numpy as np
import glob
import pickle
from pathlib import Path
import csv
import re

# define the list of boundaries
# pourcentage nuage https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
#
# # Paramétrage détection images
# sensitivity = 35
# lower_white = np.array([0, 200, 0])
# upper_white = np.array([255, 255, 255])
#
# data_folder = Path('S:/1_Photos_aeriennes/PVA_Catalogues/')
# database_catalogue = r'C:\Users\yroncin\Documents_Yoann\01_projets\02_GIS\00_fusion_PVA\00_python\database_catalogue'
# database_nuage = r'C:\Users\yroncin\Documents_Yoann\01_projets\02_GIS\00_fusion_PVA\00_python\database_catalogue_nuage'
#
# def read_listing(pickle_file):
#     with open(pickle_file, 'rb') as fp:
#         listing = pickle.load(fp, encoding='latin1')
#         return listing
#
# def write_listing(db):
#     with open(database_nuage, 'wb') as fp:
#         pickle.dump(db, fp)
#
#
# def open_image_files(catalogue, nom):
#     jpg_path = data_folder / catalogue / 'ImagesReduites' / str(nom + '.jpg')
#     jgw_path = data_folder / catalogue / 'ImagesReduites' / str(nom + '.jgw')
#     tif_path = data_folder / catalogue / str(nom + '.tif')
#     jpg_folder = jpg_path.parents[0]
#     tif_folder = tif_path.parents[0]
#     path_data = {'jpg': jpg_path, 'tif': tif_path, 'jgw': jgw_path,
#                  'jpg_folder': jpg_folder, 'tif_folder': tif_folder}
#     return path_data
#
# def myround(x, base=5):
#     return int(base * round(float(x)/base))
#
# catalogue_PVA = read_listing(database_catalogue)
#
# # pprint.pprint(catalogue_PVA)
#
# def nuage_finder():
#     file_liste = []
#     for key, value in catalogue_PVA.items():
#         catalogue_PVA[key]['nuage'] = []
#         for type, files in value['files'].items():
#             if type == 'jpeg':
#                 for i, file in enumerate(files):
#                     # print(file, key)
#                     # print(catalogue_PVA[key]['files']['jpeg'][i])
#                     # print('------------------------------------')
#
#
#                     img = open_image_files(key, file)
#                     img = str(img['jpg'])
#                     image = cv2.imread(img)
#                     try:
#                         hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#                     except:
#                         print(img)
#
#                     # find the colors within the specified boundaries and apply
#                     # the mask
#
#                     mask = cv2.inRange(hsl, lower_white, upper_white)
#                     width, height = mask.shape
#                     nbr_pixels = width * height
#
#
#                     pourcent_nuage = cv2.countNonZero(mask)/nbr_pixels *100
#                     print(cv2.countNonZero(mask)/nbr_pixels)
#                     pourcent_nuage = np.round(pourcent_nuage, 2)
#
#                     output = cv2.bitwise_and(hsl, hsl, mask=mask)
#
#                     grey_mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#                     ret, thresh = cv2.threshold(grey_mask, 2, 255, 0)
#
#                     im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#                     surf = 0
#                     for cnt in contours:
#                         if cv2.contourArea(cnt) / (image.size / 4) * 100 > 0.5:
#                             cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
#                             cv2.drawContours(output, [cnt], 0, 255, -1)
#                             # print('contour surf: ',np.round(cv2.contourArea(cnt)/(image.size/4)*100, 2))
#                             surf += cv2.contourArea(cnt)
#
#                     surf = int(np.round(surf / (image.size / 4) * 100))
#                     print(pourcent_nuage)
#                     pourcent_nuage = myround(pourcent_nuage)
#                     print(pourcent_nuage)
#                     catalogue_PVA[key]['nuage'].append(pourcent_nuage)
#
#                     print(key)
#
#                     # show the images
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     cv2.putText(image, str(pourcent_nuage) + ' % \n tous, partiel : ' + str(surf), (0, 30), font, 1, (200, 255, 155))
#                     cv2.imshow(str(pourcent_nuage) + ' % tous, partiel : ' + str(surf) + ' %', np.hstack([image, output]))
#                     cv2.waitKey(0)
#     write_listing(catalogue_PVA)
#
# # nuage_finder()
#
#
# def read_listing(pickle_file):
#     with open(pickle_file, 'rb') as fp:
#         listing = pickle.load(fp)
#         return listing
#
# catalogue_PVA = read_listing(database_nuage)
#
#
#
# regex = r"^(?P<archipel>[A-Z]{3,4})[-](?P<ile>[A-Z]{3})[-](?P<type>[A-Z]{3})[-](?P<echelle>[0-9]{4,5})[-](?P<annee>199[0-9]|198[0-9]|197[0-9]|196[0-9]|195[0-9]|200[0-9])[-](?P<numero>[0-9]{4}|[0-9]{4}[a-z]{1})[-](?P<catalogue>[C][0-9]{3}$|[C][0-9]{3}[a-z]{1}$)"
# def nom_catalogue(nom_catalogue):
#     matches = re.finditer(regex, nom_catalogue)
#     for matchNum, match in enumerate(matches):
#         if match is None:
#             raise ValueError
#         else:
#             return match.group(7)
#
# with open('nuage_report.csv', 'w', newline='') as csvout:
#     spamwriter = csv.writer(csvout, delimiter=';')
#     spamwriter.writerow(['catalogue','pourcent_nuage', 'nom_image'])
#     for key, value in catalogue_PVA.items():
#         for i, j in enumerate(value['files']['jpeg']):
#             # spamwriter.writerow([value, date[i], montant[i], code_opération[i], empty_file[i]])
#             # print(j, key, value)
#             spamwriter.writerow([nom_catalogue(j), value['nuage'][i], j])
#
#




print('Press a key to iterate over the samples')
path_images = glob.glob('./sample/*.jpg')
lower_white = np.array([0, 200, 0])
upper_white = np.array([255, 255, 255])


for img in path_images:
    image = cv2.imread(img)
    hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # find the colors within the specified boundaries and apply
    # the mask

    mask = cv2.inRange(hsl, lower_white, upper_white)
    width, height = mask.shape
    nbr_pixels = width * height

    pourcent_nuage = cv2.countNonZero(mask)/nbr_pixels *100
    pourcent_nuage = np.round(pourcent_nuage, 2)

    output = cv2.bitwise_and(hsl, hsl, mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pourcent_sup5 = pourcent_nuage
    surf = 0
    area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)/nbr_pixels*100

        if area < 0.5:
            # find areas less then 0.5 to remove small sparse features
            cv2.drawContours(output, [cnt], 0, 255, -1)
            pourcent_sup5 -= area
        else:
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)


    surf = np.round(surf, 2)

    # show the images
    font = cv2.FONT_HERSHEY_SIMPLEX
    indication = f'{pourcent_nuage} % all, {np.round(pourcent_sup5,1)} % cleaned small'
    print(indication)
    cv2.imshow(indication, np.hstack([image, output]))
    cv2.waitKey(0)

