import math
import os
import sys

import altair as alt
import folium
import geopandas as gpd
import pandas as pd
import rasterio.mask
from branca.element import Template, MacroElement
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USER-DEFINED VARIABLES AND INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Variable 'base_dir' is the project home directory where this script resides, defined by user
base_dir = r'C:\Projects\design-storm'

# Variable 'basin_shapefile_dir' is the directory where the HEC-HMS subbasin delineation .shp exists, defined by user
basin_shapefile_dir = base_dir + os.sep + r'GisData\BasinShapefiles'

# Variable 'subbasin_shapefile' is the name of the shapefile with the HEC-HMS subbasin delineations, defined by user
subbasin_shapefile = 'Trinity_CWMS_Subbasins.shp'

# Variable 'subbasin_col_name' represents the name of the column in the 'subbasin_shapefile' specified above where the
# subbasin names are stored. This needs editing by the user to exactly match the shapefile column name.
subbasin_col_name = 'Name'

# Variable 'na14_dir' is the path to na14 ams precipitation frequency .asc files, defined by user
na14_dir = base_dir + os.sep + r'GisData\PrecipNA14'

# Variable 'na14List' contains the na14 filenames (starting with 1hr up to total storm duration), defined by user
# The files in this list should coincide with the 'durations' variable below
na14List = [na14_dir + os.sep + 'tx100yr60ma_ams.asc',
            na14_dir + os.sep + 'tx100yr02ha_ams.asc',
            na14_dir + os.sep + 'tx100yr03ha_ams.asc',
            na14_dir + os.sep + 'tx100yr06ha_ams.asc',
            na14_dir + os.sep + 'tx100yr12ha_ams.asc',
            na14_dir + os.sep + 'tx100yr24ha_ams.asc',
            na14_dir + os.sep + 'tx100yr48ha_ams.asc']

# Variable 'durations' is a list of storm durations used in the analysis, defined by user
# There should be a corresponding na14 ams precipitation frequency .asc file for each duration in the list
durations = [1, 2, 3, 6, 12, 24, 48]

# Variable 'frequency' is the return period of the analysis, defined by user
frequency = '100yr'

# Variable 'output_name' is a unique identifier used to name this script's output files, defined by user
output_name = frequency

# Variable 'output_dir' is the directory where output from this script will be located
output_dir = base_dir + os.sep + r'ScriptResults\2_Prepare_NA14_Precipitation'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REPROJECT NA14 RASTERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This section, and beyond, includes variables that should generally not be changed. Thus no special editing is needed.

# Variable 'epsg_out' defines the output reference system for the subbasin file and precipitation output
epsg_out = 5070  # USA_Contiguous_Albers_Equal_Area_Conic_USGS_version in meters

# Reproject na14 precipitation rasters
print("\n***** Reprojecting NA14 precipitation rasters... *****\n")
for filename in na14List:
    with rasterio.open(filename) as source:
        transform, width, height = calculate_default_transform(source.crs, epsg_out, source.width, source.height,
                                                               *source.bounds)
        kwargs = source.meta.copy()
        kwargs.update({'crs': epsg_out, 'transform': transform, 'width': width, 'height': height})

        with rasterio.open(filename[:-4] + '_proj.asc', 'w', **kwargs) as destination:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(destination, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=transform,
                dst_crs=epsg_out,
                resampling=Resampling.bilinear)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BUILD SUBBASIN TEMPORAL PATTERNS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read in the shapefile with HEC-HMS subbasin delineations
gdf_basin = gpd.read_file(basin_shapefile_dir + os.sep + subbasin_shapefile)
gdf_basin = gdf_basin[[subbasin_col_name, 'geometry']]
gdf_basin = gdf_basin.to_crs(epsg=str(epsg_out))  # convert to defined projection

gdf_basin['Precip'] = 0

# Compute subbasin centroids
print("\n***** Computing subbasin centroids... *****\n")
gdf_basin = gdf_basin.to_crs(epsg=str(epsg_out))
gdf_basin['X'] = gdf_basin['geometry'].centroid.x
gdf_basin['Y'] = gdf_basin['geometry'].centroid.y

# Query na14 depths for each duration at each subbasin centroid
max_depths = []
all_depths = []
count = 0
subbasin_count = len(gdf_basin.index)
for index, row in gdf_basin.iterrows():
    subbasin_depths = []
    for filename in os.listdir(na14_dir):
        if frequency in filename and filename.endswith('_proj.asc'):
            with rasterio.open(na14_dir + os.sep + filename) as source:
                for value in source.sample([(row['X'], row['Y'])]):
                    subbasin_depths.append(value[0] / 1000.0)
    subbasin_depths.sort()
    if len(subbasin_depths) != len(durations):
        print('ERROR: The number of queried depths (' + str(len(subbasin_depths)) + ') does not equal the number of '
                                                                                    'specified durations (' + str(
            len(durations)) + ').')
        sys.exit('Run aborted.')
    all_depths.append(subbasin_depths)
    max_depths.append(subbasin_depths[-1])
    count = count + 1
    print('Precipitation depths queried for subbasin ' + str(count) + ' of ' + str(subbasin_count) + '.')
gdf_basin['Depths'] = all_depths
gdf_basin['Precip'] = max_depths

# Define variables/function related to the NA14 depths
nans = [float("NaN")] * durations[-1]
durations_all = range(1, durations[-1] + 1, 1)


def tp40(duration, area_mi2):  # where duration is in hrs and area is in sqmi
    reduction = 1 - math.exp(-1.1 * duration ** 0.25) + math.exp(-1.1 * (duration ** 0.25) - 0.01 * area_mi2)
    return reduction


# For each subbasin (index) in the geo-dataframe, compute the temporal pattern
print('\n***** Building a temporal pattern for each subbasin... *****\n')
temporal_patterns = []
for index, row in gdf_basin.iterrows():
    # Create a blank pandas data frame with an index for each hour of the total storm duration
    df = pd.DataFrame(list(zip(durations_all, nans)), columns=['Dur', 'Empty'])
    # Create a pandas data frame with extracted NA14 precip values; merge dataframes
    df2 = pd.DataFrame(list(zip(durations, row['Depths'])), columns=['Dur', 'Depths'])
    df3 = pd.merge(df, df2, how='left', on='Dur')
    df3.drop('Empty', axis=1, inplace=True)
    del df, df2

    # Interpolate NA14 depths for all durations
    df3.interpolate(method='linear', inplace=True)

    # Apply TP40 reductions to the NA14 depths
    df3['TP40'] = df3.apply(lambda x: tp40(x['Dur'], 400), axis=1)

    # Finalize TP40 temporal pattern; export to list
    df3['Depths_TP40'] = df3['Depths'] * df3['TP40']
    df3['Depths_Inc_TP40'] = df3['Depths_TP40'].diff()
    df3.loc[0, 'Depths_Inc_TP40'] = df3.loc[0, 'Depths_TP40']

    if durations[-1] == 6:
        depth_order = [4, 3, 5, 2, 6, 1]
    elif durations[-1] == 12:
        depth_order = [7, 6, 8, 5, 9, 4, 10, 3, 11, 2, 12, 1]
    elif durations[-1] == 24:
        depth_order = [13, 12, 14, 11, 15, 10, 16, 9, 17, 8, 18, 7,
                       19, 6, 20, 5, 21, 4, 22, 3, 23, 2, 24, 1]
    elif durations[-1] == 48:
        depth_order = [25, 24, 26, 23, 27, 22, 28, 21, 29, 20, 30, 19,
                       31, 18, 32, 17, 33, 16, 34, 15, 35, 14, 36, 13,
                       37, 12, 38, 11, 39, 10, 40, 9, 41, 8, 42, 7,
                       43, 6, 44, 5, 45, 4, 46, 3, 47, 2, 48, 1]
    elif durations[-1] == 96:
        depth_order = [49, 48, 50, 47, 51, 46, 52, 45, 53, 44, 54, 43,
                       55, 42, 56, 41, 57, 40, 58, 39, 59, 38, 60, 37,
                       61, 36, 62, 35, 63, 34, 64, 33, 65, 32, 66, 31,
                       67, 30, 68, 29, 69, 28, 70, 27, 71, 26, 72, 25,
                       73, 24, 74, 23, 75, 22, 76, 21, 77, 20, 78, 19,
                       79, 18, 80, 17, 81, 16, 82, 15, 83, 14, 84, 13,
                       85, 12, 86, 11, 87, 10, 88, 9, 89, 8, 90, 7,
                       91, 6, 92, 5, 93, 4, 94, 3, 95, 2, 96, 1]
    else:
        print('ERROR: Only total storm durations of 6, 12, 24, 48, or 96 hours are currently supported.')
        sys.exit('Run aborted.')

    df3['Depth_Order'] = depth_order
    df3.sort_values(by=['Depth_Order'], inplace=True)
    temporal_pattern = df3['Depths_Inc_TP40'].round(2).tolist()
    del df3
    temporal_patterns.append(temporal_pattern)

gdf_basin['Temporal'] = temporal_patterns
gdf_basin.to_csv(output_dir + os.sep + '2a_Subbasins_Precip_Vector_' + output_name + '.csv')
header = [subbasin_col_name, 'Temporal']
gdf_basin.to_csv(output_dir + os.sep + '2b_Subbasins_TemporalPattern_' + output_name + '.csv', columns=header)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE A SUBBASIN TEMPORAL MAP (HTML) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n***** Creating a map... *****\n")
gdf_centroid = gdf_basin.copy()
gdf_centroid = gdf_centroid[[subbasin_col_name, 'Temporal', 'geometry']]

# Convert list columns to string so that the geo-dataframe can be written to a shapefile
gdf_basin['Depths'] = gdf_basin['Depths'].apply(str)
gdf_basin['Temporal'] = gdf_basin['Temporal'].apply(str)
gdf_basin.to_file(output_dir + os.sep + '2a_Subbasins_Precip_Vector_' + output_name + '.shp')

# Create a centroid geo-dataframe from the subbasin geo-dataframe
gdf_centroid['geometry'] = gdf_basin['geometry'].centroid
gdf_centroid = gdf_centroid.to_crs(epsg='4326')

# Simplify basin geometry
gdf_basin['geometry'] = gdf_basin['geometry'].simplify(75)
gdf_basin = gdf_basin.to_crs(epsg='4326')

# Identify center coordinates for map
bounds = gdf_basin.total_bounds
lon_low = bounds[2]
lon_high = bounds[0]
lat_low = bounds[1]
lat_high = bounds[3]

map_center_lon = (lon_low + lon_high) / 2.0
map_center_lat = (lat_low + lat_high) / 2.0

# Create a folium map object
mapa = folium.Map([map_center_lat, map_center_lon],
                  zoom_start=8,
                  tiles='cartodbpositron',
                  control_scale=True
                  )

# Add basin layer to map
style = {'color': 'gray', 'fillColor': 'gray', 'fillOpacity': 0.2, 'weight': 1.0}
folium.GeoJson(data=gdf_basin,
               name="Basin",
               style_function=lambda x: style,
               ).add_to(mapa)

# Add choropleth layer of subbasins to map
folium.Choropleth(
    geo_data=gdf_basin,
    name="Subbasin Choropleth",
    data=gdf_basin,
    columns=[subbasin_col_name, "Precip"],
    key_on="feature.properties." + subbasin_col_name,
    fill_color="Reds",
    fill_opacity=0.8,
    line_opacity=0.1,
    legend_name=frequency + str(durations[-1]) + 'hr Precipitation Depth [in.]',
    smooth_factor=0,
    Highlight=True,
    line_color="#0000",
    show=True,
    overlay=True
).add_to(mapa)

# Add hover functionality to basin layer
style_function = lambda x: {'fillColor': '#ffffff',
                            'color': '#000000',
                            'fillOpacity': 0.001,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color': '#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}
hover = folium.features.GeoJson(
    data=gdf_basin,
    name="Subbasin Overlay",
    style_function=style_function,
    control=True,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=[subbasin_col_name, "Precip"],
        aliases=['Subbasin Name:', frequency + str(durations[-1]) + ' Precip Depth:'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
)
mapa.add_child(hover)

# Add subbasin centroid layer to map
feature_group = folium.FeatureGroup(name=frequency + str(durations[-1]) + 'hr Temporal Pattern')
for index, row in gdf_centroid.iterrows():
    data_list = row['Temporal']
    df_plot = pd.DataFrame(list(zip(durations_all, data_list)), columns=['Duration', 'Temporal'])

    # Create temporal pattern chart for popup feature
    bar_plot = alt.Chart(df_plot, title=row[subbasin_col_name] + ': ' + frequency + str(durations[-1]) +
                         'hr Temporal Pattern'
                         ).mark_bar().encode(
        x=alt.X('Duration', axis=alt.Axis(title='Duration (hr)', labelFontSize=14)),
        y=alt.Y('Temporal', axis=alt.Axis(title='Precipitation Depth (in.)', labelFontSize=14)),
    ).properties(width=600, height=400)

    popup = folium.Popup(max_width=1000)
    folium.VegaLite(bar_plot, width=650, height=450).add_to(popup)

    # Create marker for each subbasin centroid
    folium.CircleMarker(
        location=(row['geometry'].y, row['geometry'].x),
        radius=3,
        popup=popup,
        tooltip=frequency + str(durations[-1]) + 'hr Temporal Pattern',
        color='black',
        fill_color='black',
        fill_opacity=0.01,
        opacity=0.1
    ).add_to(feature_group)
feature_group.add_to(mapa)

mapa.keep_in_front(hover, feature_group)

# Add various background tile layers
folium.TileLayer('cartodbdark_matter', name="cartodb dark", control=True).add_to(mapa)
folium.TileLayer('openstreetmap', name="open street map", control=True, opacity=0.4).add_to(mapa)
folium.TileLayer('stamenterrain', name="stamen terrain", control=True, opacity=0.6).add_to(mapa)
folium.TileLayer('stamenwatercolor', name="stamen watercolor", control=True, opacity=0.6).add_to(mapa)
folium.TileLayer('stamentoner', name="stamen toner", control=True, opacity=0.6).add_to(mapa)

# Add a layer controller
folium.LayerControl(collapsed=True).add_to(mapa)

# Create a legend using HTML and JavaScript
template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<body>


<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:18px; left: 6px; bottom: 40px;'>

<div class='legend-title'>Legend</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:gray;opacity:0.5;'></span>Basin</li>
    <li><span style='background:darkred;opacity:0.4;'></span>Precipitation Depths</li>
    <li><span style='background:black;opacity:0.7;'></span>Temporal Patterns</li>


  </ul>
</div>
</div>

</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)
mapa.get_root().add_child(macro)

# Save map to html
mapa.save(output_dir + os.sep + '2c_Subbasins_Precip_Map_' + output_name + '.html')

print("****************************************************************")
print("Temporal patterns have been created for each subbasin.")
print("The script results have been saved in: " + output_dir)
print("Run Complete.")
print("****************************************************************")
