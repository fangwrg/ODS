import csv
import os
import shutil as shutilb
import sys
from datetime import datetime, timedelta

import altair as alt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from branca.element import Template, MacroElement
from pydsstools._lib.x64.py37.core_heclib import TimeSeriesContainer, squeeze_file
from pydsstools.heclib.dss import HecDss
from rasterio import features, shutil, mask
from rasterstats import zonal_stats
from shapely import affinity

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USER-DEFINED VARIABLES AND INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Variable 'base_dir' is the project directory where this script resides, needs editing by user
base_dir = r'C:\Projects\design-storm'

# Variable 'hms_model_dir' is a directory for the HMS model, needs editing by user
hms_model_dir = base_dir + os.sep + r'HmsProject\Trin_HMS'

# Variable 'hms_project' is the project name of the HMS model, needs editing by user
hms_project = "CWMS_Trinity_River"

# Variable 'hms_run' is the name of the HMS simulation created for design storm optimization
hms_run = "DesignStorm"

# Variable 'ctrl_spec_time_interval' is the HEC-HMS Control Specifications Time Interval, i.e. use '15MIN' or '1HOUR'
ctrl_spec_time_interval = '1HOUR'

# Variable 'hms_met_model' is the name of the HMS meteorological model (specified hyetograph) for design storm runs
hms_met_model = "DesignStorm.met"

# Variable 'start_time_str' is a string for the start time of HMS simulation, the user needs to edit and keep a format
# of 'ddmmmyyyy,HHMMSS'.
start_time_str = "01Jan2000,000000"

# 'dss_in' is the path and name of the input HEC-HMS DSS file of subbasin specified hyetographs
dss_in = hms_model_dir + os.sep + r'data\hyetographs.dss'

# 'dss_out' is the path and name of the output HEC-HMS DSS file with computation results
# The output DSS file by default is named after the HMS run, with hyphens and spaces replaced with underscores
dss_out = hms_model_dir + os.sep + hms_run.replace("-", "_").replace(" ", "_") + '.dss'

# Variable 'junction_shapefile' is the path and name of the junction shapefile, exported from HEC-HMS, defined by user
junction_shapefile = base_dir + os.sep + r'GisData\BasinShapefiles\Trinity_CWMS_Junctions.shp'

# Variable 'junction_col_name' represents the name of the column in 'junction_shapefile' specified above where the
# junction names are stored. This needs editing by the user to exactly match the shapefile column name.
junction_col_name = 'Name'

# Variable 'frequency' is the return period of the analysis, defined by user
frequency = '100yr'  # update for each frequency run!!!!!!

# Variable 'duration' is the total storm duration of the analysis, defined by user
duration = '48hr'

# Variable 'basin_name' is the name of the study basin, used to label output, defined by user
basin_name = "Trinity"

# Variable 'output_name' is a unique identifier used to name this script's output files, defined by user
# 'output_name' should change with each script 4 run, otherwise previously computed results may be overwritten
output_name = frequency + duration + "_flow"

# Variable 'output_dir' is the directory where output from this script will be located
output_dir = base_dir + os.sep + r'ScriptResults\4_Finalize_HMS_Elliptical_Design_Storm_Runs'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ USER-DEFINED INPUT NEEDED FROM SCRIPTS 1, 2, and 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Variable 'ellipses_shapefile' is the path and file name of the DAR shapefile (output from script 1), defined by user
ellipses_shapefile = base_dir + os.sep + r'ScriptResults\1_Prepare_DAR_Ellipses\1a_Ellipses_Vector_dar48hr.shp'

# Variable 'na14_raster' is the path and name of the precipitation frequency .asc file, defined by user
# This is the total duration .asc file that was projected in script 2.
na14_raster = base_dir + os.sep + r'GisData\PrecipNA14\tx100yr48ha_ams_proj.asc'

# Variable 'subbasin_precip_shp' is the path and name of the subbasin shapefile (output from script 2), defined by user
subbasin_precip_shp = base_dir + os.sep + \
                       r'ScriptResults\2_Prepare_NA14_Precipitation\2a_Subbasins_Precip_Vector_100yr.shp'

# Variable 'subbasin_precip_csv' is the path and name of the subbasin csv (output from script 2), defined by user
subbasin_precip_csv = base_dir + os.sep + \
                    r'ScriptResults\2_Prepare_NA14_Precipitation\2a_Subbasins_Precip_Vector_100yr.csv'

# Variable 'subbasin_col_name' represents the name of the column in the 'subbasin_precip_shp' specified above where the
# subbasin names are stored. This needs editing by the user to exactly match the shapefile column name.
subbasin_col_name = 'Name'

# Variable 'csv_in' is the path and name of the optimization script output that includes a list of all the
# junctions and their optimized storm location and angle
csv_in = base_dir + os.sep + r'ScriptResults\3_Run_HMS_Elliptical_Design_Storm_Optimization' \
                           r'\3b_Trinity_XYThetaOpt_100yr48hr_flow.csv '

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIALIZATION BASED ON PRIOR USER INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# This section, and beyond, includes variables that should generally not be changed. Thus no special editing is needed.

# Variable 'hyetograph_time_step' is the interval of the hyetograph in hours.
hyetograph_time_step = 1  # script currently setup to run at 1 hour hyetograph time step only

# Variable 'theta0' is the initial orientation of the design storm, '0' means the storm is positioned horizontally.
# Degrees are measured counter clockwise from the positive x-axis
theta0 = 0

# Variable 'hms_install_dir' is the installation directory for HEC-HMS.exe
hms_install_dir = base_dir + os.sep + r'Software\HEC-HMS-4.8'

# Variable 'software_dir' is a directory for various software
software_dir = base_dir + os.sep + 'Software'

# Variable 'jdk_install_dir' is the installation directory for Java (jdk)
# The jdk is needed to call HMS headlessly
jdk_install_dir = software_dir + os.sep + 'jdk-11.0.10'

# Variable 'jython_install_exe' is the installation directory for Jython
# Jython is needed to call HMS headlessly
jython_install_exe = software_dir + os.sep + r'jython-2.7.2\bin\jython.exe'

# 'temp_dir' is a directory where temporary files are stored
temp_dir = software_dir + os.sep + 'TempFiles'

# Variable 'csv_out' is the path and name of this script's output csv summary file
csv_out = output_dir + os.sep + '4a_' + basin_name + 'FinalizedRuns' + output_name + '.csv'

# Read in subbasin files from script 2 and use the centroid coordinate to set the initial storm center
df_basin = pd.read_csv(subbasin_precip_csv, converters={'Temporal': eval})
gdf_basin = gpd.read_file(subbasin_precip_shp)
gdf_basin['Temporal'] = df_basin['Temporal']
gdf_basin['Precip'] = 0
subbasin_list = gdf_basin[subbasin_col_name].tolist()

basin_dissolved = gdf_basin.copy()
basin_dissolved = basin_dissolved.dissolve(by='Precip')
basin_centroid = basin_dissolved['geometry'].centroid

x0 = float(basin_centroid.x)
y0 = float(basin_centroid.y)
bounds = basin_dissolved.total_bounds

del basin_dissolved

# 'jython_script' is the name of script file used to run HMS automatically.
jython_script = temp_dir + os.sep + 'HMScompute.py'

# 'start_time_num' is the numerical form of the simulation start time.
start_time_num = datetime.strptime(start_time_str, '%d%b%Y,%H%M%S')

# 'hms_metmodel_backup' is a copy of the hms met model. The met model is copied over before each run to avoid a bug that
# sometimes deletes precip gages from the met model file when hms is called repeatedly
shutilb.copy(hms_model_dir + os.sep + hms_met_model, temp_dir)
hms_metmodel_backup = temp_dir + os.sep + hms_met_model

# ~~~~~~~~~~~~~~~~~~~ PREPARATION: WRITE AN HMSCOMPUTE.PY JYTHON SCRIPT FILE ~~~~~~~~~~~~~~~~~~~

# Write a Jython script file that simulates the prepared simulation in the target HMS model
file = open(jython_script, "w")
file.write("from hms.model import Project" + "\n")
file.write("from hms import Hms" + "\n")
file.write("myProject = Project.open('" + hms_model_dir + os.sep + hms_project + ".hms')" + "\n")
file.write("myProject.computeRun('" + hms_run + "')" + "\n")
file.write("myProject.close()" + "\n")
file.write("Hms.shutdownEngine()" + "\n")
file.close()
print("\n***** Preparing HMScompute.py jython script for HMS simulation run... *****\n")

# ~~~~~~~~~~~~~~~~~~~~~ PREPARATION: WRITE AN HMSCOMPUTE.BAT FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Write an HMScompute.bat file that calls HMS and executes the HMScompute.py Jython file
file = open(temp_dir + os.sep + "HMScompute.bat", "w")
file.write("@echo off" + "\n")
file.write('\n')
file.write('set "HMS_HOME=' + hms_install_dir + '"' + '\n')
file.write(r'set "PATH=%HMS_HOME%\bin\gdal;%PATH%"' + '\n')
file.write(r'set "GDAL_DRIVER_PATH=%HMS_HOME%\bin\gdal\gdalplugins"' + '\n')
file.write(r'set "GDAL_DATA=%HMS_HOME%\bin\gdal\gdal-data"' + '\n')
file.write(r'set "PROJ_LIB=%HMS_HOME%\bin\gdal\projlib"' + '\n')
file.write(r'set "CLASSPATH=%HMS_HOME%\hms.jar;%HMS_HOME%\lib\*"' + '\n')
file.write('set "JAVA_HOME=' + jdk_install_dir + '"' + '\n')
file.write('\n')
file.write(r'%HMS_HOME%\jre\bin\java -Djava.library.path=%HMS_HOME%\bin;%HMS_HOME%\bin\gdal;%HMS_HOME%\bin\hdf; '
           r'org.python.util.jython ' + jython_script + '\n')
file.close()
print("\n***** Preparing .bat file that executes HMScompute.py jython script... *****\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RE-CREATE THE TOTAL OPTIMIZED STORM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The DAR ellipses are rotated and transposed to the previously optimized location, then rasterized
# Then the underlying NA14 precipitation point frequency depths are queried and multiplied by the DAR raster
# Then the average, total storm precipitation for each subbasin is calculated

peak_flow = []  # create list 'peak_flow' to store peak flows after each loop
x_opt_list = []  # create list to store previously optimized longitudes
y_opt_list = []  # create list to store previously optimized latitudes
theta_opt_list = []  # create list to store previously optimized storm angles
junction_list = []  # create list to store junction names
depths_max_hr = []  # create list to store NA14 96hr depths for each location

# Open the output csv from the optimization script and extract some optimized values
print("\n***** Extracting optimized storm locations... *****\n")
with open(csv_in, 'r') as f:
    next(f)
    for column in csv.reader(f):
        x_opt_list.append(float(column[2])),
        y_opt_list.append(float(column[3])),
        theta_opt_list.append(float(column[4])),
        junction_list.append(column[0])

# Loop through all the junctions with their previously optimized storm locations
counter = 0
for i in range(len(junction_list)):
    # Assign the current junction name
    junction = junction_list[i]
    # Assign the optimized values for the current junction
    x_opt = x_opt_list[i]
    y_opt = y_opt_list[i]
    theta_opt = theta_opt_list[i]

    print("\n***** For " + junction + ", storm to be moved to \n   " + str(x_opt) + ", " + str(y_opt) + ", at " + str(
        theta_opt) + " degrees. *****\n")

    print("\n***** Rotating and translating storm to previously optimized location... *****\n")
    # Read in the ellipses shapefile from script 1
    gdf_ellipse = gpd.read_file(ellipses_shapefile)
    gdf_ellipse.reset_index(drop=True)

    # Rotate the ellipses
    ellipse_rotated = gdf_ellipse.copy()
    for index, row in ellipse_rotated.iterrows():
        rotated = shapely.affinity.rotate(row['geometry'], angle=theta0 - theta_opt)
        ellipse_rotated.loc[index, 'geometry'] = rotated

    # Translate the ellipses
    ellipse_translated = ellipse_rotated.copy()
    for index, row in ellipse_translated.iterrows():
        translated = shapely.affinity.translate(row['geometry'], xoff=x_opt - x0, yoff=y_opt - y0)
        ellipse_translated.loc[index, 'geometry'] = translated

    # Mask the reprojected na14 precipitation raster based on rotated/translated ellipses
    with rasterio.open(na14_raster) as template:
        out_precip, out_transform = rasterio.mask.mask(dataset=template, shapes=ellipse_translated.geometry, crop=True)
        no_data = template.nodata
        out_precip[out_precip == no_data] = 0
        precip_meta = template.meta.copy()
        precip_meta.update({"driver": "AAIGrid",
                           "height": out_precip.shape[1],
                           "width": out_precip.shape[2],
                           "transform": out_transform,
                           "nodata": 0,
                           "dtype": "int32"})

        # Get storm center depth
        for center_depth in template.sample([(x_opt, y_opt)]):
            depths_max_hr.append(center_depth[0] / 1000.0)

    # Filename of the clipped na14 raster that will be created
    na14_clipped = temp_dir + os.sep + 'na14_clip.asc'

    # If it exists, delete a previously masked na14 raster file
    if os.path.isfile(na14_clipped):
        try:
            rasterio.shutil.delete(na14_clipped)
        except BaseException as error:
            print('An exception occurred: {}'.format(error))

    with rasterio.open(na14_clipped, "w", **precip_meta) as destination:
        destination.write(out_precip)

    # Filename of the transposed dar raster that will be created
    dar_transposed = temp_dir + os.sep + 'dar_transposed.tif'

    # Burn the rotated/translated ellipse features into a raster
    dar_meta = precip_meta.copy()
    dar_meta.update({"driver": "GTiff",
                    "nodata": 0,
                    "dtype": "float64"})

    with rasterio.open(dar_transposed, 'w+', **dar_meta) as dar_transposed_out:
        out_array = dar_transposed_out.read(1)
        shapes = ((geom, value) for geom, value in zip(ellipse_translated.geometry, ellipse_translated.dar))
        burned = features.rasterize(shapes=shapes, fill=0, out=out_array, transform=dar_transposed_out.transform)
        dar_transposed_out.write_band(1, burned)

    # Multiply the masked na14 precip by the transposed dar raster, divide by 1000 to convert to decimal inches
    with rasterio.open(dar_transposed) as source:
        dar = source.read(1, masked=True)

    with rasterio.open(na14_clipped) as source:
        masked_precip = source.read(1, masked=True)

    reduced_precip = masked_precip * dar / 1000.00
    reduced_precip_meta = dar_meta.copy()

    # Filename of the reduced storm raster that will be created
    storm_final = output_dir + os.sep + '4b_Storm_' + junction + '.tif'

    with rasterio.open(storm_final, 'w+', **reduced_precip_meta) as out_raster:
        out_raster.write_band(1, reduced_precip)

    # Calculate zonal statistics to determine the average precipitation for each subbasin
    print("\n***** Calculating zonal statistics for the total, average precipitation at each subbasin... *****\n")
    with rasterio.open(storm_final, 'r+') as source:
        array = source.read(1)
        affine = source.transform
    stats = zonal_stats(gdf_basin, array, affine=affine, nodata=-9999, stats='mean', band=1)
    mean_values = []
    for entry in stats:
        mean_value = entry.get('mean')
        if mean_value is None:
            mean_value = 0
        mean_values.append(round(mean_value, 3))

    # Delete precipitation rasters created during this iteration
    try:
        rasterio.shutil.delete(na14_clipped)
        rasterio.shutil.delete(dar_transposed)
    except BaseException as error:
        print('An exception occurred: {}'.format(error))

    # Add subbasin average precip values to geo dataframe
    gdf_basin['Precip'] = mean_values
    gdf_basin['Precip'] = gdf_basin['Precip'].fillna(0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPARE HMS HYETOGRAPH INPUT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Write specified hyetograph data to dss
    start_date_time = (start_time_num + timedelta(hours=1)).strftime("%d%b%Y %H:%M:%S").upper()
    temporal_length = len(gdf_basin['Temporal'][0])
    depths_all = []
    for index, row in gdf_basin.iterrows():
        depths = []
        for dur in range(temporal_length):
            depths.append(row['Precip'] * row['Temporal'][dur] / sum(row['Temporal']))
        depths_all.append(depths)
        # Use pydsstools library to convert Pandas dataframe to DSS
        pathname = "//" + row[subbasin_col_name].upper() + "/PRECIP-INC//" + str(hyetograph_time_step).upper() \
                   + "HOUR//"
        tsc = TimeSeriesContainer()
        tsc.pathname = pathname
        tsc.startDateTime = start_date_time
        tsc.numberValues = temporal_length
        tsc.units = "IN"
        tsc.type = "PER-CUM"
        tsc.interval = hyetograph_time_step  # hours
        tsc.values = depths

        fid = HecDss.Open(dss_in)
        fid.deletePathname(tsc.pathname)
        fid.put_ts(tsc)
        fid.close()

    gdf_basin['IncTS'] = depths_all
    squeeze_file(dss_in)

    # Create a directory to store the specified hyetographs DSS files for each final run
    hyetograph_dir = os.path.dirname(dss_in) + os.sep + junction + '_' + output_name
    if not os.path.isdir(hyetograph_dir):
        os.mkdir(hyetograph_dir)

    # Remove previous DSS files if they exist; this ensures that they are updated in this iteration
    base_name = os.path.basename(dss_in)
    if os.path.isfile(hyetograph_dir + os.sep + base_name):
        os.remove(hyetograph_dir + os.sep + base_name)

    # Copy the specified hyetographs DSS file to the created directory
    shutilb.copy(dss_in, hyetograph_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SIMULATE HMS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n***** Created the specified hyetographs DSS file; Calling HEC-HMS... *****\n")
    os.chdir(temp_dir)
    # Copy over the HMS met model file to avoid a bug that sometimes removes precip gages from the met model
    os.remove(hms_model_dir + os.sep + hms_met_model)
    shutilb.copy(hms_metmodel_backup, hms_model_dir)
    sys_command = ".\\HMScompute.bat"
    os.system(sys_command)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXTRACT PEAK FLOW ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Extract the junction peak flow from the DSS file holding model output
    print("\n***** Reading the output DSS file... *****\n")
    pathname = "//" + junction.upper() + "/FLOW//" + ctrl_spec_time_interval.upper() + "/RUN:" + hms_run + "/"

    fid = HecDss.Open(dss_out)
    time_series = fid.read_ts(pathname)
    values = time_series.values
    times = np.array(time_series.pytimes)
    try:
        max_val = max(values)
    except TypeError:
        print("The specified 'dss_out' file and 'pathname' cannot be found.")
        sys.exit(1)
    peak_flow.append(max_val)

    # Extract incremental precipitation values for each subbasin
    incremental_ts_list = []
    for subbasin in subbasin_list:
        subbasin_pathname = "//" + subbasin + "/PRECIP-INC//" + ctrl_spec_time_interval + "/RUN:" + hms_run + "/"
        precip_time_series = fid.read_ts(subbasin_pathname)
        precip_values = precip_time_series.values
        if precip_values is None:
            precip_values = np.zeros(len(times))
        incremental_ts_list.append(precip_values.tolist())
    fid.close()

    squeeze_file(dss_out)

    gdf_basin['IncTS'] = incremental_ts_list

    # Check HEC-HMS run log for errors; write error to script error log if found
    hms_run_log = hms_model_dir + os.sep + os.path.splitext(os.path.basename(dss_out))[0] + ".log"
    script_error_log = output_dir + os.sep + "0_HEC-HMS_ERROR_LOG_" + hms_run + ".txt"
    if os.path.exists(hms_run_log):
        with open(hms_run_log, 'r') as run_log:
            for line in run_log:
                if line.startswith("ERROR"):
                    if os.path.exists(script_error_log):
                        write_mode = 'a'
                    else:
                        write_mode = 'w'
                    with open(script_error_log, write_mode) as error_log:
                        error_log.write(line)
                        error_log.write("The below simulation results for " + junction + " are invalid: \n")
                        error_log.write(str(row) + "\n")
                        error_log.write("\n")
                    print("\n***** HEC-HMS ERROR!!!!! Check error log in output directory... *****\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CREATE A TOTAL STORM MAP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("\n***** Creating a map... *****\n")

    # Simplify basin geometry for file size efficiency
    gdf_basin_map = gdf_basin.copy()
    gdf_basin_map = gdf_basin_map[[subbasin_col_name, 'Precip', 'geometry']]
    gdf_basin_map['geometry'] = gdf_basin_map['geometry'].simplify(150)

    # Identify center coordinates for map
    gdf_basin_map = gdf_basin_map.to_crs(epsg='4326')
    bounds_lat_lon = gdf_basin_map.total_bounds
    lon_low = bounds_lat_lon[2]
    lon_high = bounds_lat_lon[0]
    lat_low = bounds_lat_lon[1]
    lat_high = bounds_lat_lon[3]
    map_center_lon = (lon_low + lon_high) / 2.0
    map_center_lat = (lat_low + lat_high) / 2.0

    # Create a folium map object
    mapa = folium.Map([map_center_lat, map_center_lon],
                      zoom_start=8,
                      tiles='cartodbpositron',
                      control_scale=True,
                      zoom_control=True,
                      dragging=True,
                      scrollWheelZoom=True
                      )

    # Add basin layer to map
    style = {'color': 'gray', 'fillColor': 'gray', 'fillOpacity': 0.2, 'weight': 1.0}
    folium.GeoJson(data=gdf_basin_map,
                   name="Basin",
                   # popup=gdf_basin_map[subbasin_col_name],
                   style_function=lambda x: style,
                   ).add_to(mapa)

    # Add choropleth layer of subbasins to map
    folium.Choropleth(
        geo_data=gdf_basin_map,
        name="Subbasin Choropleth",
        data=gdf_basin_map,
        columns=[subbasin_col_name, "Precip"],
        key_on="feature.properties." + subbasin_col_name,
        fill_color="Reds",
        fill_opacity=0.8,
        line_opacity=0.1,
        legend_name="Avg. Precipitation Depths: " + junction,
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
        data=gdf_basin_map,
        name="Subbasin Overlay",
        style_function=style_function,
        control=True,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=[subbasin_col_name, "Precip"],
            aliases=['Subbasin Name:', 'Avg Precip Depth:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )
    mapa.add_child(hover)
    del gdf_basin_map

    # Create subbasin centroid layer and cumulative precipitation plots
    gdf_centroid = gdf_basin.copy()
    gdf_centroid = gdf_centroid[[subbasin_col_name, 'IncTS', 'geometry']]
    gdf_centroid['geometry'] = gdf_basin['geometry'].centroid
    gdf_centroid = gdf_centroid.to_crs(epsg='4326')
    gdf_centroid['geometry'] = gdf_centroid['geometry'].simplify(0.001)

    feature_group = folium.FeatureGroup(name='Subbasin Centroids')
    for index, row in gdf_centroid.iterrows():
        data_list = row['IncTS']
        df_plot = pd.DataFrame(list(zip(times, data_list)), columns=['Time', 'IncTS'])
        # Resample time series data to 1 hour resolution for file size efficiency
        df_plot['Time'] = pd.to_datetime(df_plot['Time'], format='%Y-%m-%d %H%M%S')
        df_plot = df_plot.set_index('Time')
        df_plot = df_plot.resample('1H').mean()
        df_plot.reset_index(inplace=True)
        df_plot = df_plot.rename(columns={'index': 'Time'})

        # Create temporal pattern chart for popup feature
        line_plot = alt.Chart(df_plot, title=frequency + duration + ' Incremental Precipitation: ' + row[subbasin_col_name]) \
            .mark_line().encode(x=alt.X('Time', axis=alt.Axis(title='Date', format='%b-%d', labelOverlap=True, labelSeparation=20)),
                                y=alt.Y('IncTS', axis=alt.Axis(title='Incremental Precipitation (in.)')),
                                color=alt.value('#FF2D00')
                                ).properties(width=600, height=300)

        popup = folium.Popup(max_width=1000)
        folium.VegaLite(line_plot, width=700, height=350).add_to(popup)

        # Create marker for each subbasin centroid
        folium.CircleMarker(
            location=(row['geometry'].y, row['geometry'].x),
            radius=3,
            popup=popup,
            tooltip='Click for precipitation plot...',
            color='black',
            fill_color='black',
            fill_opacity=0.01,
            opacity=0.1
        ).add_to(feature_group)
    feature_group.add_to(mapa)

    # Add optimized storm outline to map
    storm_outline = ellipse_translated.copy()
    storm_outline['geometry'] = storm_outline['geometry'].simplify(90)
    storm_outline = storm_outline.iloc[[0]]

    style2 = {'color': 'black', 'fillColor': 'gray', 'fillOpacity': 0.20, 'weight': 2.0}
    folium.GeoJson(data=storm_outline,
                   name="Optimized Storm Position",
                   tooltip=frequency + duration + " Optimized Storm",
                   style_function=lambda x: style2,
                   ).add_to(mapa)

    # Prepare junction layer for map
    gdf_junction = gpd.read_file(junction_shapefile)
    gdf_junction = gdf_junction.to_crs(epsg='4326')
    junction_index = None
    for index, row in gdf_junction.iterrows():
        if row[junction_col_name] == junction:
            junction_index = index
            break

    junction_lon = gdf_junction.loc[junction_index, 'geometry'].centroid.x
    junction_lat = gdf_junction.loc[junction_index, 'geometry'].centroid.y

    # Create hydrograph popup plot
    df_plot2 = pd.DataFrame(list(zip(times, values)), columns=['Time', 'Flow'])
    line_plot = alt.Chart(df_plot2, title='Optimized ' + frequency + duration + ' Hydrograph: ' + junction) \
        .mark_line().encode(x=alt.X('Time', axis=alt.Axis(title='Date', format='%b-%d', labelOverlap=True, labelSeparation=20)),
                            y=alt.Y('Flow', axis=alt.Axis(title='Flow (cfs)')),
                            ).properties(width=600, height=300)

    # Add plot to popup
    popup = folium.Popup(max_width=1000)
    folium.VegaLite(line_plot, width=700, height=350).add_to(popup)

    # Add junction layer with popup plot to map
    feature_group2 = folium.FeatureGroup(name='Junction: ' + junction)
    folium.Marker(
        location=[junction_lat, junction_lon],
        popup=popup,
        tooltip=junction,
        icon=folium.Icon(icon="cloud", color="blue")
    ).add_to(feature_group2)
    feature_group2.add_to(mapa)

    # Add various background tile layers
    folium.TileLayer('cartodbdark_matter', name="cartodb dark", control=True).add_to(mapa)
    folium.TileLayer('openstreetmap', name="open street map", control=True, opacity=0.4).add_to(mapa)
    folium.TileLayer('stamenterrain', name="stamen terrain", control=True, opacity=0.6).add_to(mapa)
    folium.TileLayer('stamenwatercolor', name="stamen watercolor", control=True, opacity=0.6).add_to(mapa)
    folium.TileLayer('stamentoner', name="stamen toner", control=True, opacity=0.6).add_to(mapa)

    # Add a layer controller
    folium.LayerControl(collapsed=True).add_to(mapa)

    mapa.keep_in_front(hover, feature_group, feature_group2)

    # Create a legend using HTML and JavaScript
    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <body>


    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:18px; left: 6px; bottom: 40px;'>

    <div class='legend-title'>Optimized Storm Legend</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li><span style='background:black;opacity:0.7;'></span>Optimized Storm</li>
        <li><span style='background:darkred;opacity:0.4;'></span>Precipitation Depths</li>



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
    mapa.save(output_dir + os.sep + '4c_Storm_Map_' + junction + '_' + output_name + '.html')

    counter = counter + 1

    print("\n***** Junction " + str(counter) + " of " + str(len(junction_list)) + " complete! *****\n")

# Write an out csv file with the storm parameters and peak flow for each junction
header = ['Junction', 'PeakFlow', 'X', 'Y', 'Theta', 'CenterDepth']
rows = zip(junction_list, peak_flow, x_opt_list, y_opt_list, theta_opt_list, depths_max_hr)
with open(csv_out, "w") as h:
    writer = csv.writer(h, lineterminator='\n')
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

# Clear 'temp_dir'
for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutilb.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s from TempDir. Reason: %s' % (file_path, e))

print("****************************************************************")
print("Finalized Design Storm runs have been made for " + str(len(junction_list)) + " junctions.")
print("The script results have been saved in: " + output_dir)
print("Run Complete.")
print("****************************************************************")
