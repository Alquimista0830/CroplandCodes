import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, polygonize
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
from tqdm import trange

# ------------------------- Main Direction -------------------------
def calculate_main_direction_from_pca(geometry):
    # Processing MultiPolygon
    if geometry.geom_type == 'MultiPolygon':
        largest_poly = max(geometry.geoms, key=lambda p: p.area)
        coords = np.array(largest_poly.exterior.coords)
    elif geometry.geom_type == 'Polygon':
        coords = np.array(geometry.exterior.coords) 
    else:
        raise ValueError(f"Not support polygon type: {geometry.geom_type}")
    # PCA
    centered = coords - np.mean(coords, axis=0)
    pca = PCA(n_components=2)
    pca.fit(centered)
    main_direction_vector = pca.components_[0]
    angle_rad = np.arctan2(main_direction_vector[1], main_direction_vector[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 180, main_direction_vector 


def calculate_main_direction_from_bounding_box(geometry):
    # Bounding box
    if geometry.geom_type == 'LineString':
        min_rotated_rectangle = geometry.minimum_rotated_rectangle
    else:  # Polygon/MultiPolygon
        min_rotated_rectangle = geometry.minimum_rotated_rectangle
    
    coords = np.array(min_rotated_rectangle.exterior.coords)
    vectors = [coords[i + 1] - coords[i] for i in range(4)]
    lengths = [np.linalg.norm(v) for v in vectors]
    longest_edge_index = np.argmax(lengths)
    longest_edge_vector = vectors[longest_edge_index]
    
    # angle calculation
    angle_rad = np.arctan2(longest_edge_vector[1], longest_edge_vector[0])
    angle_deg = np.degrees(angle_rad) % 180  # standard 0-180 
    return angle_deg, longest_edge_vector, min_rotated_rectangle


# ------------------------- Regularity -------------------------
def convert_line_to_polygon_and_calculate_area(geometry):
    if geometry.geom_type == 'LineString':
        polygon = Polygon(geometry.coords)
    elif geometry.geom_type == 'Polygon':
        polygon = geometry
    else:
        return np.nan, None
    return polygon.area, polygon


def calculate_regularity(perimeter, area):
    if area <= 0 or perimeter <= 0:
        return np.nan
    return 2 * np.log(perimeter / 4) / np.log(area)


def calculate_regularity_bounding_box(area, bounding_box_area):
    if bounding_box_area <= 0:
        return np.nan
    return area / bounding_box_area


# ------------------------- Contiguity -------------------------
def convert_line_to_polygon(geometry):
    if geometry.is_empty or not geometry.is_valid:
        return None
    # Polygon processing
    if geometry.geom_type == 'Polygon':
        return geometry
    # MultiPolygon processing
    elif geometry.geom_type == 'MultiPolygon':
        return geometry
    # LineString processing
    elif geometry.geom_type == 'LineString':
        return Polygon(geometry.coords)
    # MultiLineString processing
    elif geometry.geom_type == 'MultiLineString':
        polygons = list(polygonize(geometry))
        if polygons:
            return unary_union(polygons)
        else:
            return None
    else:
        return None


def calculate_contiguity_index(areas):
    if len(areas) == 0:  
        return np.nan  
    S_min = np.min(areas)
    S_max = np.max(areas)
    if S_min == S_max:  
        return np.nan 
    return (np.log(areas) - np.log(S_min)) / (np.log(S_max) - np.log(S_min))


# ------------------------- Main -------------------------
def main():
    # User input
    shapefile_path = r'.shp' # input shp file
    threshold_distance = 10.0 # buffering distance (m)
    output_folder = r'' # output dir

    # read Shapefile
    gdf = gpd.read_file(shapefile_path)

    # make output dir
    os.makedirs(output_folder, exist_ok=True)

    # add FID col
    gdf['FID'] = gdf.index

    # ------------------------- Main direction -------------------------
    pca_shapefile = []
    bbox_shapefile = []
    rect_shapefile = []
    pca_angles = []
    bbox_angles = []

    for idx, geometry in tqdm(enumerate(gdf.geometry), total= len(gdf)):
        # calculate main direction
        pca_angle_deg, pca_main_direction_vector = calculate_main_direction_from_pca(geometry)
        bbox_angle_deg, bbox_longest_edge_vector, min_rotated_rectangle = calculate_main_direction_from_bounding_box(geometry)

        line_center = geometry.centroid
        center_x, center_y = line_center.x, line_center.y

        bbox_coords = np.array(min_rotated_rectangle.exterior.coords)
        longest_edge_length = np.linalg.norm(bbox_coords[0] - bbox_coords[1])
        arrow_length = longest_edge_length / 2

        if pca_angle_deg > 180:
            pca_angle_deg -= 180
        if bbox_angle_deg > 180:
            bbox_angle_deg -= 180

        pca_direction_vector_scaled = np.array([np.cos(np.radians(pca_angle_deg)), np.sin(np.radians(pca_angle_deg))]) * arrow_length
        bbox_direction_vector_scaled = np.array([np.cos(np.radians(bbox_angle_deg)), np.sin(np.radians(bbox_angle_deg))]) * arrow_length

        pca_arrow_start = Point(center_x, center_y)
        pca_arrow_end = Point(center_x + pca_direction_vector_scaled[0], center_y + pca_direction_vector_scaled[1])
        pca_shapefile.append(LineString([pca_arrow_start, pca_arrow_end]))

        bbox_arrow_start = Point(center_x, center_y)
        bbox_arrow_end = Point(center_x + bbox_direction_vector_scaled[0], center_y + bbox_direction_vector_scaled[1])
        bbox_shapefile.append(LineString([bbox_arrow_start, bbox_arrow_end]))

        rect_shapefile.append(min_rotated_rectangle)

        pca_angles.append(pca_angle_deg)
        bbox_angles.append(bbox_angle_deg)

    gdf['PCA_main_direction'] = pca_angles
    gdf['BoundingBox_main_direction'] = bbox_angles

    # save results
    pca_gdf = gpd.GeoDataFrame({'FID': gdf['FID'], 'geometry': pca_shapefile}, crs=gdf.crs)
    bbox_gdf = gpd.GeoDataFrame({'FID': gdf['FID'], 'geometry': bbox_shapefile}, crs=gdf.crs)
    rect_gdf = gpd.GeoDataFrame({'FID': gdf['FID'], 'geometry': rect_shapefile}, crs=gdf.crs)

    pca_gdf.to_file(os.path.join(output_folder, 'PCA_main_direction.shp'))
    bbox_gdf.to_file(os.path.join(output_folder, 'BoundingBox_main_direction.shp'))
    rect_gdf.to_file(os.path.join(output_folder, 'BoundingBox.shp'))

    # ------------------------- Regularity -------------------------
    gdf['perimeter'] = gdf.geometry.length  # perimeter
    gdf['area'], gdf['polygon_geometry'] = zip(*gdf.geometry.apply(convert_line_to_polygon_and_calculate_area))
    gdf['regularity'] = gdf.apply(lambda row: calculate_regularity(row['perimeter'], row['area']), axis=1)

    gdf['bounding_box_area'] = gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle.area)
    gdf['regularity_bounding_box'] = gdf.apply(lambda row: calculate_regularity_bounding_box(row['area'], row['bounding_box_area']), axis=1)

    regularity_gdf = gdf[['FID', 'polygon_geometry', 'regularity', 'regularity_bounding_box']].rename(columns={'polygon_geometry': 'geometry'})
    regularity_gdf = gpd.GeoDataFrame(regularity_gdf, geometry='geometry', crs=gdf.crs)

    regularity_gdf = regularity_gdf.rename(columns={
        'regularity': 'regul',
        'regularity_bounding_box': 'regul_bbox'
    })
    regularity_gdf.to_file(os.path.join(output_folder, 'Field_Regularity.shp'))

    # ------------------------- Contiguity -------------------------
    print(gdf.geometry.geom_type.value_counts())
    gdf['geometry'] = gdf['geometry'].apply(convert_line_to_polygon)
    # check converted
    print(gdf.geometry.geom_type.value_counts())
    gdf = gdf.dropna(subset=['geometry'])
    if len(gdf) == 0:
        raise ValueError("There is no valid geometry data available to calculate contiguity!")
    gdf['area'] = gdf.geometry.area

    # Checking buffer operations
    gdf['buffer'] = gdf.geometry.buffer(threshold_distance)
    if gdf['buffer'].is_empty.any():
        print("Warning: Some geometries have empty buffers, which may cause the contiguous calculation to fail")

    unioned_geometry = unary_union(gdf['buffer'])
    unioned_gdf = gpd.GeoDataFrame(geometry=[unioned_geometry], crs=gdf.crs)
    unioned_gdf = unioned_gdf.explode(index_parts=True).reset_index(drop=True)
    unioned_gdf['area'] = unioned_gdf.geometry.area

    # Check if there is a valid area
    if len(unioned_gdf) == 0:
        print("WARNING: No valid contiguous region was generated!")
        unioned_gdf['contiguity_index'] = np.nan  # Add a null value column
    else:
        areas = unioned_gdf['area'].values
        unioned_gdf['contiguity_index'] = calculate_contiguity_index(areas)

    # save results
    unioned_gdf['FID'] = unioned_gdf.index  # Add FID column
    unioned_gdf = unioned_gdf.rename(columns={'contiguity_index': 'contiguity'})
    unioned_gdf.to_file(os.path.join(output_folder, 'Contiguous_Regions.shp'))

    # ------------------------- Output Excel -------------------------
    # Merge all results into raw data
    print('Output Excel')
    gdf_with_contiguity = gpd.sjoin(gdf, unioned_gdf, how='left', predicate='intersects')
    gdf_with_contiguity['FID'] = gdf_with_contiguity.index

    print("Saving Excel results...")
    excel_output_path = os.path.join(output_folder, 'Field_Shape_Indicators.xlsx')

    main_directions_df = gdf[['FID', 'PCA_main_direction', 'BoundingBox_main_direction']]
    print("Column names for the Regularity Table:", regularity_gdf.columns.tolist())  
    regularity_df = regularity_gdf[['FID', 'regul', 'regul_bbox']]
    print("The column name of the contiguous table:", unioned_gdf.columns.tolist()) 
    contiguity_df = unioned_gdf[['FID', 'area', 'contiguity']]

    
    def save_to_excel_in_chunks(df, output_path, sheet_name, chunk_size=1000000):
        num_chunks = len(df) // chunk_size + 1
        for i in trange(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk_df = df.iloc[start:end]
            if not chunk_df.empty:
                chunk_output_path = output_path.replace('.xlsx', f'_{sheet_name}_part{i+1}.xlsx')
                chunk_df.to_excel(chunk_output_path, sheet_name=sheet_name, index=False)
                print(f"Save chunk {i+1} to {chunk_output_path}")


    save_to_excel_in_chunks(main_directions_df, excel_output_path, sheet_name='Main_Directions')

    save_to_excel_in_chunks(regularity_df, excel_output_path, sheet_name='Regularity') 

    save_to_excel_in_chunks(contiguity_df, excel_output_path, sheet_name='Contiguity')

    print(f"All results have been saved to: {output_folder}")

if __name__ == "__main__":
    main()