/****************************************************************************
 * Basin-average daily NEX-GDDP-CMIP6 series for the Tlawng River Basin
 * (Google Earth Engine Code Editor script)
 *
 * Part of: "A Hybrid Hydrology-AI Framework with Terrain-Based Inundation
 * Modeling for Probabilistic Flood Hazard-Exposure Assessment under CMIP6
 * Scenarios" (Debbarma, Bandyopadhyay, Bhadra) - Module 1.
 *
 * Produces one CSV per (variable, scenario, GCM) with columns
 *   Date, <value column>
 * which is the format read by the Python scripts (files in Raw_Projections/
 * and the Raw_Modeled column in Historical/).
 *
 * HOW TO USE
 *  1. Upload "Basin Shapefile/Tlawng_Catchment.shp" (+ .shx .dbf .prj .cpg)
 *     to your GEE Assets and set CATCHMENT_ASSET below to its asset ID.
 *  2. Choose VARIABLE, SCENARIO, period and MODELS.
 *  3. Run, then start the export tasks from the "Tasks" tab.
 *     CSVs are written to the Google Drive folder EXPORT_FOLDER.
 *
 * Units written:
 *  pr     kg m-2 s-1 x 86400       -> mm/day       (Mean_PR_mm_day)
 *  rsds   W m-2 x 24 / 1000        -> kWh/m2/day   (Mean_RSDS_kWh_m2_day)
 *  tasmax K - 273.15               -> degC         (Mean_Tmax_C)
 *  tasmin K - 273.15               -> degC         (Mean_Tmin_C)
 ****************************************************************************/

/**** 1. Settings ****/
var CATCHMENT_ASSET = 'users/sagarbd33/Tlawng_Catchmnet';   // <<< your asset ID
var VARIABLE = 'pr';            // 'pr' | 'rsds' | 'tasmax' | 'tasmin'
var SCENARIO = 'historical';    // 'historical' | 'ssp245' | 'ssp585'

// Reference period 1986-2014 for 'historical'; 2015-2099 for the SSPs.
// filterDate end is exclusive.
var START = (SCENARIO === 'historical') ? '1986-01-01' : '2015-01-01';
var END   = (SCENARIO === 'historical') ? '2015-01-01' : '2100-01-01';

var EXPORT_FOLDER = 'CMIP6_Areal';
var SCALE = 25000;   // metres; NEX-GDDP-CMIP6 native grid is 0.25 deg

// All NEX-GDDP-CMIP6 GCMs (34). CESM2, CESM2-WACCM and IITM-ESM provide no
// tasmax/tasmin; HadGEM3-GC31-MM provides no ssp245 rsds. Missing combinations
// export an empty CSV, which can be deleted.
var MODELS = [
  'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM',
  'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'CanESM5',
  'EC-Earth3', 'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-CM4', 'GFDL-ESM4',
  'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'IITM-ESM',
  'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM',
  'MIROC-ES2L', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0',
  'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1', 'UKESM1-0-LL'
];
// To export only the top-5 GCMs of a variable, replace MODELS, e.g. for pr:
// var MODELS = ['KACE-1-0-G', 'EC-Earth3-Veg-LR', 'INM-CM5-0',
//               'HadGEM3-GC31-LL', 'CNRM-CM6-1'];

/**** 2. Variable definitions ****/
var DEF = {
  pr:     {prefix: 'PR',     column: 'Mean_PR_mm_day',
           convert: function (x) { return x.multiply(86400); }},
  rsds:   {prefix: 'RSDS',   column: 'Mean_RSDS_kWh_m2_day',
           convert: function (x) { return x.multiply(0.024); }},
  tasmax: {prefix: 'TASMAX', column: 'Mean_Tmax_C',
           convert: function (x) { return x.subtract(273.15); }},
  tasmin: {prefix: 'TASMIN', column: 'Mean_Tmin_C',
           convert: function (x) { return x.subtract(273.15); }}
};
var def = DEF[VARIABLE];
var catchment = ee.FeatureCollection(CATCHMENT_ASSET).geometry();

/**** 3. Daily basin-average series for one GCM ****/
function arealSeries(modelName) {
  var col = ee.ImageCollection('NASA/GDDP-CMIP6')
    .filter(ee.Filter.eq('model', modelName))
    .filter(ee.Filter.eq('scenario', SCENARIO))
    .filterDate(START, END)
    .select(VARIABLE);

  return col.map(function (img) {
    var mean = ee.Number(img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: catchment,
      scale: SCALE,
      maxPixels: 1e13
    }).get(VARIABLE));
    var props = {'Date': img.date().format('YYYY-MM-dd')};
    props[def.column] = def.convert(mean);
    return ee.Feature(null, props);
  });
}

/**** 4. One export task per GCM ****/
MODELS.forEach(function (modelName) {
  var fc = ee.FeatureCollection(arealSeries(modelName));
  var name = def.prefix + '_Areal_' + SCENARIO + '_' + modelName;
  Export.table.toDrive({
    collection: fc,
    description: name.replace(/[^A-Za-z0-9_-]/g, '_'),
    folder: EXPORT_FOLDER,
    fileNamePrefix: name,
    fileFormat: 'CSV',
    selectors: ['Date', def.column]      // drops 'system:index' and '.geo'
  });
});

print('Export tasks created for', MODELS.length, 'GCMs:', VARIABLE, SCENARIO, START, 'to', END);
