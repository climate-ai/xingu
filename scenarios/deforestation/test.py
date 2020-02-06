# The layer we use as test labels
gfc2014 = ee.Image('UMD/hansen/global_forest_change_2015');
Map.addLayer(gfc2014, {
  bands: ['loss', 'treecover2000', 'gain'],
  max: [1, 255, 1]
}, 'forest cover, loss, gain');

