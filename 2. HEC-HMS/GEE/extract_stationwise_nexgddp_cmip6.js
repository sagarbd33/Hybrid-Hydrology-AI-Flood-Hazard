/****************************************************************************
 * Station-wise (nearest grid cell) daily NEX-GDDP-CMIP6 series at the 12 IWRD
 * rain-gauge locations of the Tlawng River Basin (Google Earth Engine).
 *
 * Part of: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
 * Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
 * Scenarios" (Debbarma, Bandyopadhyay, Bhadra) - Module 2 forcing.
 *
 * Output CSV (one per variable/scenario/GCM):
 *   Date, Aizawl, Sialsuk, ..., Serchhip      (format of Data/GCM_Stationwise/)
 * Stations that fall in the same 0.25-degree cell receive identical values.
 *
 * GCMs used in the paper (top-ranked per variable, Module 1):
 *   pr -> KACE-1-0-G,  tasmax -> FGOALS-g3,  tasmin -> INM-CM4-8
 *
 * HOW TO USE: fill in STATIONS (lon, lat in decimal degrees), choose VARIABLE,
 * SCENARIO and MODEL, run, then start the export task in the Tasks tab.
 ****************************************************************************/

/**** 1. Settings ****/
var VARIABLE = 'pr';           // 'pr' | 'tasmax' | 'tasmin'
var SCENARIO = 'ssp245';       // 'historical' | 'ssp245' | 'ssp585'
var MODEL    = 'KACE-1-0-G';
var START = (SCENARIO === 'historical') ? '1984-01-01' : '2015-01-01';
var END   = (SCENARIO === 'historical') ? '2015-01-01' : '2100-01-01';   // exclusive
var EXPORT_FOLDER = 'CMIP6_Stationwise';

// <<< Enter the gauge coordinates (longitude, latitude) >>>
var STATIONS = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Aizawl'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Sialsuk'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Neihbawi'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Kolasib'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Thingdawl'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Lunglei'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Hnahthial'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Haulawng'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Mamit'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'kawrtethawveng'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Zawlnuam'}),
  ee.Feature(ee.Geometry.Point([0, 0]), {name: 'Serchhip'})
]);
var NAMES = ['Aizawl', 'Sialsuk', 'Neihbawi', 'Kolasib', 'Thingdawl', 'Lunglei',
             'Hnahthial', 'Haulawng', 'Mamit', 'kawrtethawveng', 'Zawlnuam', 'Serchhip'];

/**** 2. Unit conversion ****/
function convert(img) {
  if (VARIABLE === 'pr') { return img.multiply(86400); }        // kg m-2 s-1 -> mm/day
  return img.subtract(273.15);                                  // K -> degC
}

/**** 3. Daily values at the station points (nearest 0.25-degree cell) ****/
var col = ee.ImageCollection('NASA/GDDP-CMIP6')
  .filter(ee.Filter.eq('model', MODEL))
  .filter(ee.Filter.eq('scenario', SCENARIO))
  .filterDate(START, END)
  .select(VARIABLE);

var daily = col.map(function (img) {
  var vals = convert(img).reduceRegions({
    collection: STATIONS, reducer: ee.Reducer.first(), scale: 27830
  });
  var dict = ee.Dictionary.fromLists(vals.aggregate_array('name'), vals.aggregate_array('first'));
  return ee.Feature(null, dict.set('Date', img.date().format('YYYY-MM-dd')));
});

/**** 4. Export ****/
var prefix = {pr: 'PR', tasmax: 'TASMAX', tasmin: 'TASMIN'}[VARIABLE];
var name = prefix + '_Stationwise_' + SCENARIO + '_' + MODEL;
Export.table.toDrive({
  collection: ee.FeatureCollection(daily),
  description: name.replace(/[^A-Za-z0-9_-]/g, '_'),
  folder: EXPORT_FOLDER,
  fileNamePrefix: name,
  fileFormat: 'CSV',
  selectors: ['Date'].concat(NAMES)
});
