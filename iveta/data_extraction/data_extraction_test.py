import os
import ee

ee.Initialize()

## Input imagery is Sentinel-2 data

# Function to mask clouds using the Sentinel-2 QA band
# @param ee.Image image Sentinel-2 image
# @return ee.Image cloud masked Sentinel-2 image

def maskS2clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
      .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

  return image.updateMask(mask).divide(10000)

def maskS2val(image):
  return image.divide(10000)


## Load Sentinel-2 TOA reflectance data and mask clouds
sentinel2_collection = ee.ImageCollection('COPERNICUS/S2') \
                    .filterDate('2016-05-01', '2016-12-01') \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)) \
                    .map(maskS2val)
#.map(maskS2clouds)

## Use these bands for prediction
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
#sentinel2_collection = sentinel2_collection.select(bands)
bands_10m = ['B2', 'B3', 'B4', 'B8']
bands_20m = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

## Resample bands to 10m per pixel
def resample(image):
  img_res = image.select(bands_10m)
  for i in range(len(bands_20m)):
    band = bands_20m[i]
    img_res = img_res.addBands(image.select(band) \
      .resample('bilinear') \
      .reproject( \
        crs=image.select(bands_20m[i]).projection().crs(), \
        scale=10))
  return img_res

sentinel2_resampled = sentinel2_collection.map(resample)

## Create the composite image
sentinel2_composite = sentinel2_resampled.min()
#.median()
#.reduce(ee.Reducer.percentile([5]))

## Load land use dataset labels
landuse_dataset = ee.FeatureCollection('users/ivetarott/landuse_data')

## Add centre of region point geometry
## This function adds the feature's geometry as a property.
def addGeometry(feature):
  x = feature.getNumber('location_x')
  y = feature.getNumber('location_y')
  return feature.setGeometry(ee.Geometry.Point([x, y]))

## Map the geometry setting function over the FeatureCollection.
landuse_dataset = landuse_dataset.map(addGeometry)

## Add a new property of random number from 0 to 1 to split data
landuse_dataset = landuse_dataset.randomColumn('random', 1)
#print(landuse_dataset.first())

# Split the landuse data into training and validation datasets
split_1 = 0.0
split_2 = 0.6
split_val_1 = 0.6
split_val_2 = 0.8
split_test_1 = 0.8
split_test_2 = 1.0

coef = 0.0002 #used for splitting in batches with coef*100 % of the full data set each
i=0

landuse_train = []
landuse_val = []
landuse_test = []

i = round(split_1/coef)
while(i*coef<split_2):
  landuse_train.append(landuse_dataset.filter(ee.Filter.gte('random', i*coef)) \
                    .filter(ee.Filter.lt('random', (i+1)*coef)))
  i+=1
  
i = round(split_val_1/coef)
while(i*coef<split_val_2):
  landuse_val.append(landuse_dataset.filter(ee.Filter.gte('random', i*coef)) \
                    .filter(ee.Filter.lt('random', (i+1)*coef)))
  i+=1
  
i = round(split_test_1/coef)
while(i*coef<split_test_2):
  landuse_test.append(landuse_dataset.filter(ee.Filter.gte('random', i*coef)) \
                    .filter(ee.Filter.lt('random', (i+1)*coef)))
  i+=1

print(len(landuse_train))
print(len(landuse_val))
print(len(landuse_test))


landuse_train_idx = landuse_dataset.filter(ee.Filter.gte('random', split_1)) \
                    .filter(ee.Filter.lt('random', split_2))
landuse_val_idx = landuse_dataset.filter(ee.Filter.gte('random', split_val_1)) \
                    .filter(ee.Filter.lt('random', split_val_2))
landuse_test_idx = landuse_dataset.filter(ee.Filter.gte('random', split_test_1)) \
                    .filter(ee.Filter.lt('random', split_test_2))

#print(landuse_train_idx.size())
#print(landuse_val_idx.size())
#print(landuse_test_idx.size())

## Export index files to Google Drive
task = ee.batch.Export.table.toDrive( \
  collection = landuse_train_idx, \
  description = '00_train_index', \
  fileFormat = 'CSV', \
  selectors = ['public_id'], \
  folder = 'EE_Data_index_v02')
#task.start()

task = ee.batch.Export.table.toDrive( \
  collection = landuse_val_idx, \
  description = '00_validate_index', \
  fileFormat = 'CSV', \
  selectors = ['public_id'], \
  folder = 'EE_Data_index_v02')
#task.start()

task = ee.batch.Export.table.toDrive( \
  collection = landuse_test_idx, \
  description = '00_test_index', \
  fileFormat = 'CSV', \
  selectors = ['public_id'], \
  folder = 'EE_Data_index_v02')
#task.start()

# Remap the landuse cover percentage to integer labels
# Remap on subsets of landuse data for better parallelization
percentage_labels = ['0%', '1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%','9%','10-19%', '20-29%', '30-39%', '40-49%', '50-59%', '60-69%', '70-79%', '80-89%', '90-100%']
integer_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
label_names = ['vegetation_elementstree_element_cover_label',
                  'vegetation_elementsshrub_element_cover_label',
                  'vegetation_elementspalm_element_cover_label',
                  'vegetation_elementsbamboo_element_cover_label',
                  'vegetation_elementscrop_element_cover_label',
                  'infrastructure_elementshouse_element_cover_label',
                  'infrastructure_elementsother_buildings_element_cover_label',
                  'infrastructure_elementspaved_road_element_cover_label',
                  'infrastructure_elementsunpaved_road_element_cover_label',
                  'water_bodieslake_water_cover_label',
                  'water_bodiesriver_water_cover_label',
                  'total_water_bodies_cover_label']
                  
for i in range(len(landuse_train)):
  #landuse_train[i] = landuse_train[i].remap(percentage_labels, integer_labels, 'vegetation_elementstree_element_cover_label')
  for j in range(len(label_names)):
    landuse_train[i] = landuse_train[i].remap(percentage_labels, integer_labels, \
                        label_names[j])

for i in range(len(landuse_val)):
  #landuse_val[i] = landuse_val[i].remap(percentage_labels, integer_labels, 'vegetation_elementstree_element_cover_label')
  for j in range(len(label_names)):
    landuse_val[i] = landuse_val[i].remap(percentage_labels, integer_labels, \
                        label_names[j])
    
for i in range(len(landuse_test)):
  #landuse_test[i] = landuse_test[i].remap(percentage_labels, integer_labels,  'vegetation_elementstree_element_cover_label')
  for j in range(len(label_names)):
    landuse_test[i] = landuse_test[i].remap(percentage_labels, integer_labels, \
                        label_names[j])

#print(landuse_train[0].first())
#print(landuse_val[0].first())
#print(landuse_test[0].first())

## Overlay the points on the imagery to get training, validation and test
## labeled image patches of 70x70 m2 (7x7 pixels) & export the data


for i in range(len(landuse_test)):
    sentinel2_patches = sentinel2_composite.neighborhoodToArray(ee.Kernel.square(3))
    testing = sentinel2_patches.reduceRegions(collection=landuse_test[i], \
                                    scale=10, \
                                    reducer='first', \
                                    tileScale=1)
    task = ee.batch.Export.table.toDrive(collection=testing, \
                                    selectors=['public_id']+bands+label_names, \
                                    description='test_patches_'+"{0:04}".format(i), \
                                    fileFormat='TFRecord', \
                                    folder='EE_Data_testing_v02')
    task.start()
    
