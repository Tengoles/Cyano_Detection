from geetools import batch
import ee
ee.Initialize()

region = ee.Geometry.Polygon([[-55.166078494092936,-34.86505490701676], 
                            [-55.01913635542106,-34.86505490701676],
                            [-55.01913635542106,-34.75258121210841],
                            [-55.166078494092936,-34.75258121210841],
                            [-55.166078494092936,-34.86505490701676]])

dataset = ee.ImageCollection('COPERNICUS/S3/OLCI').filterDate('2020-03-16', '2020-03-30')

dataset = dataset.filterBounds(region)

dataset = dataset.select(['Oa07_radiance', 'Oa08_radiance', 'Oa10_radiance', 
                            'Oa11_radiance','Oa12_radiance', 'Oa18_radiance'])

#help(batch.Export.ImageCollection.toDrive)  # See help for function

tasks = batch.Export.imagecollection.toDrive(dataset, 'Satellite_data', region=region, scale=300, verbose=True)