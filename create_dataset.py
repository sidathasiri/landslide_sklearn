import pandas as pd
from numpy import column_stack
from sklearn.preprocessing import normalize

landslides = pd.read_csv("./datasets/landslides.csv")
slopes = pd.read_csv("./datasets/slope.csv")
landform = pd.read_csv("./datasets/landform.csv")
landuse = pd.read_csv("./datasets/landuse.csv")
vegetation = pd.read_csv("./datasets/vegetation.csv")
drainage_density = pd.read_csv("./datasets/drainage_density.csv")
overburden = pd.read_csv("./datasets/overburden.csv")
catchment_area = pd.read_csv("./datasets/catchment_area.csv")
elevation = pd.read_csv("./datasets/elevation.csv")

landslides = landslides.replace(255, 0)
landslides = landslides.replace([1, 2, 3, 4, 5, 6, 7, 8], 1)

print("before extract")
print(slopes.shape)
print(landform.shape)
print(landuse.shape)
print(vegetation.shape)
print(drainage_density.shape)
print(overburden.shape)
print(catchment_area.shape)
print(elevation.shape)

def extract_data_points(layer):
    set1 = layer.ix[:499, :800]
    set2 = layer.ix[500:, :]

    set1R = set1.values.reshape(500*800, 1)
    set2R = set2.values.reshape(500*1600, 1)

    return pd.concat([pd.DataFrame(set1R), pd.DataFrame(set2R)], axis=0)

for rows in range(81):
    landslides.loc[-1] = [0 for x in range(1539)]
    landslides.index = landslides.index + 1

landslides = landslides.sort_index()

for cols in range(1539, 1600):
    landslides[cols] = [0 for x in range(1000)]

slope_data = extract_data_points(slopes)
landform_data = extract_data_points(landform)
landslides_data = extract_data_points(landslides)
landuse_data = extract_data_points(landuse)
vegetation_data = extract_data_points(vegetation)
drainage_density_data = extract_data_points(drainage_density)
overburden_data = extract_data_points(overburden)
catchment_area_data = extract_data_points(catchment_area)
elevation_data = extract_data_points(elevation)

print("After extract")
print(slope_data.shape)
print(landform_data.shape)
print(landslides_data.shape)
print(landuse_data.shape)
print(vegetation_data.shape)
print(drainage_density_data.shape)
print(overburden_data.shape)
print(catchment_area_data.shape)
print(elevation_data.shape)

# print("slope\n", slope_data.head())
# print("landform\n", landform_data.head())

inputData = pd.concat([slope_data, landform_data, landuse_data, vegetation_data, drainage_density_data, overburden_data, catchment_area_data, elevation_data], axis=1)
#inputData = normalize(inputData)
#inputData = pd.DataFrame(inputData, columns=["slope", "landform", "landuse", "vegetation"])
# print("input\n", inputData.head())

inputData.columns = ["slope", "landform", "landuse", "vegetation", "drainage", "overburden", "catchment_area", "elevation"]
inputData.to_csv("input_data.csv", index=False)
landslides_data.to_csv("outputs.csv", index=False)


print("done")