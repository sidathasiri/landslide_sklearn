"""

This file will concatenate all the factors to a single CSV
All the required factors needed to be included, should be added to the concat function and to the columns list to create the csv

"""

import pandas as pd
from numpy import column_stack
from sklearn.preprocessing import normalize

#maps containing data each factor
landslides = pd.read_csv("./datasets/landslides.csv")
slopes = pd.read_csv("./datasets/slope.csv")
landform = pd.read_csv("./datasets/landform.csv")
landuse = pd.read_csv("./datasets/landuse.csv")
vegetation = pd.read_csv("./datasets/vegetation.csv")
drainage_density = pd.read_csv("./datasets/drainage_density.csv")
overburden = pd.read_csv("./datasets/overburden.csv")
catchment_area = pd.read_csv("./datasets/catchment_area.csv")
elevation = pd.read_csv("./datasets/elevation.csv")
distance = pd.read_csv("./datasets/distance.csv")

print("All files loaded...")

#encode landslide cells
landslides = landslides.replace(255, 0)
landslides = landslides.replace([1, 2, 3, 4, 5, 6, 7, 8], 1)

#removing unwanted cells
def extract_data_points(layer):
    set1 = layer.ix[:499, :800]
    set2 = layer.ix[500:, :]

    set1R = set1.values.reshape(500*800, 1)
    set2R = set2.values.reshape(500*1600, 1)

    return pd.concat([pd.DataFrame(set1R), pd.DataFrame(set2R)], axis=0)

#filling voids in landslide map
for rows in range(81):
    landslides.loc[-1] = [0 for x in range(1539)]
    landslides.index = landslides.index + 1

landslides = landslides.sort_index()

for cols in range(1539, 1600):
    landslides[cols] = [0 for x in range(1000)]

#pre-processed data to be added to CSV
slope_data = extract_data_points(slopes)
landform_data = extract_data_points(landform)
landslides_data = extract_data_points(landslides)
landuse_data = extract_data_points(landuse)
vegetation_data = extract_data_points(vegetation)
drainage_density_data = extract_data_points(drainage_density)
overburden_data = extract_data_points(overburden)
catchment_area_data = extract_data_points(catchment_area)
elevation_data = extract_data_points(elevation)
distance_data = extract_data_points((distance))

print("Data extraction complete...")

#concatenate all data to a single dataframe
#add required factors to be included in input data set
inputData = pd.concat([slope_data,
                       landform_data,
                       landuse_data,
                       overburden_data,
                       vegetation_data,
                       drainage_density_data,
                       catchment_area_data,
                       elevation_data,
                       distance_data
                       ], axis=1)

#set the column names in csv
#add the column names of concaternating factors added above
inputData.columns = ["slope",
                     "landform",
                     "landuse",
                     "overburden",
                     "vegetation",
                     "drainage_density",
                     "catchment",
                     "elevation",
                     "distance_to_streams"]

inputData.to_csv("input_data.csv", index=False)
landslides_data.to_csv("outputs.csv", index=False)


print("Datasets created!")
