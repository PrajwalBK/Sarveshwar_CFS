"""
Calibrate bbox-to-image-coords calculation using actual detection data.

This script:
1. Loads expected image coordinates from CSV (GPS -> Image)
2. Loads bbox data from crops metadata
3. Finds the optimal calculation method
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.bbox_to_image_coords import (
    calculate_image_coords_from_bbox,
    calibrate_bbox_method,
    find_optimal_percentage_offsets
)
from app.bbox_to_image_coords_advanced import (
    calculate_image_coords_advanced,
    find_optimal_linear_combination,
    calculate_with_linear_model
)


def parse_expected_coords(csv_data: str) -> Dict[str, Tuple[float, float]]:
    """Parse CSV data with expected image coordinates."""
    expected = {}
    lines = csv_data.strip().split('\n')
    reader = csv.DictReader(lines)
    for row in reader:
        spot = row['Parking Spot']
        x = float(row['Image X'])
        y = float(row['Image Y'])
        expected[spot] = (x, y)
    return expected


def load_crops_metadata(folder_path: str) -> List[Dict]:
    """Load crops metadata from a folder."""
    metadata_path = Path(folder_path) / "crops_metadata.json"
    if not metadata_path.exists():
        return []
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def match_crops_to_spots(
    crops: List[Dict],
    expected_coords: Dict[str, Tuple[float, float]],
    folder_name: str,
    spot_mapping: Optional[Dict[int, str]] = None
) -> List[Tuple[Dict, str, Tuple[float, float]]]:
    """
    Match crops to spots using manual mapping or automatic matching.
    
    Args:
        crops: List of crop metadata dicts
        expected_coords: Dict mapping spot names to (x, y) coordinates
        folder_name: Name of the folder (for manual mapping)
        spot_mapping: Optional dict mapping track_id to spot name
    
    Returns:
        List of (crop_dict, spot_name, expected_coords) tuples
    """
    matches = []
    
    for crop in crops:
        bbox = crop.get('bbox_original', crop.get('bbox', []))
        if len(bbox) != 4:
            continue
        
        track_id = crop.get('track_id')
        
        # Use manual mapping if provided
        if spot_mapping and track_id in spot_mapping:
            spot_name = spot_mapping[track_id]
            if spot_name in expected_coords:
                matches.append((crop, spot_name, expected_coords[spot_name]))
                continue
        
        # Fallback to automatic matching based on bbox position
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        
        # Find closest expected coordinate
        min_dist = float('inf')
        best_spot = None
        best_coords = None
        
        for spot, coords in expected_coords.items():
            dist = np.sqrt((center_x - coords[0])**2 + (center_y - coords[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_spot = spot
                best_coords = coords
        
        # Only include if reasonably close (within 500 pixels)
        if best_spot and min_dist < 500:
            matches.append((crop, best_spot, best_coords))
    
    return matches


def main():
    # Expected coordinates from user's CSV data
    csv_data = """Parking Spot,Image X,Image Y
DD048,1026.95288718,719.29684610
DD047,1026.95403674,719.29686679
DD046,1026.95518566,719.29688746
DD045,1026.95633396,719.29690812
DD044,1026.95748164,719.29692877
DD043,1026.95862868,719.29694941
DD042,1026.95977511,719.29697004
DD041,1026.96092089,719.29699066
DD040,1026.96206608,719.29701126
DD039,1026.96321065,719.29703186
DD038,1026.96435459,719.29705244
DD037,1026.96549791,719.29707301
DD036,1026.96664061,719.29709357
DD035,1026.96778269,719.29711412
DD034,1026.96892424,719.29713466
DD033,1026.97006509,719.29715519
DD032,1026.97120532,719.29717570
DD031,1026.97234493,719.29719621
DD030,1026.97348393,719.29721670
DD029,1026.97462230,719.29723719
DD028,1026.97576006,719.29725766
DD027,1026.97689740,719.29727812
DD026,1026.97803394,719.29729857
DD025,1026.97916986,719.29731901
DD024,1026.98030517,719.29733944
DD023,1026.99081071,719.29752845
DD022,1026.99193524,719.29754869
DD021,1026.99305874,719.29756890
DD020,1026.99418123,719.29758910
DD019,1026.99530335,719.29760929
DD018,1026.99642467,719.29762947
DD017,1026.99754520,719.29764963
DD016,1026.99866493,719.29766978
DD015,1026.99978387,719.29768991
DD014,1027.00090178,719.29771003
DD013,1027.00201914,719.29773013
DD012,1027.00313570,719.29775023
DD011,1027.00425147,719.29777030
DD010,1027.00536586,719.29779036
DD009,1027.00648004,719.29781040
DD008,1027.00759343,719.29783044
DD007,1027.00870581,719.29785045
DD006,1027.00981763,719.29787046
DD005,1027.01092866,719.29789045
DD004,1027.01203891,719.29791043
DD003,1027.01314837,719.29793039
DD002,1027.01425706,719.29795034
DD001,1027.01536423,719.29797027
YARD182,1309.70342341,724.99316104
YARD181,1574.67938943,730.59497319
YARD180,1340.56614563,726.00356202
YARD179,1205.74588332,723.36517310
YARD178,1553.43669318,732.97262394
YARD177,1112.81324549,722.95723020
YARD176,965.68498413,719.58443602
YARD175,1589.85519057,725.05129529
YARD174,-10723.77153216,562.96024684
YARD173,-151.33977401,702.05230774
YARD172,1577.39623589,727.76122561
YARD171,2069.25760326,735.81811507
YARD170,1422.21189972,725.69168105
YARD169,1580.71688477,728.38949612
YARD168,1815.33121740,732.38709239
YARD167,1462.37531439,726.59002567
YARD166,1577.99542466,728.59159304
YARD165,1739.15471763,731.37936736
YARD164,1484.30035684,727.09321134
YARD163,1576.49295077,728.70314721
YARD162,1696.49707687,730.79750302
YARD161,1498.87729951,727.42775614
YARD160,1578.15942734,728.81912104
YARD159,1452.49653601,726.66562105
YARD158,1511.07458950,727.69758935
YARD157,1577.09231197,728.86113456
YARD156,1467.87254067,726.97533268
YARD155,1518.65667372,727.87287175
YARD154,1576.31211915,728.89185177
YARD153,1478.82199192,727.19918805
YARD152,1524.54275802,728.00894664
YARD151,1577.41037955,728.94497431
YARD150,1487.67482928,727.38017933
YARD149,1530.56027653,728.14075123
YARD148,1576.76002643,728.96034189
YARD147,1496.04524775,727.54827054
YARD146,1534.29145014,728.22767785
YARD145,1468.67229847,727.07798935
YARD144,1502.10399994,727.67238289
YARD143,1537.39636408,728.30001426
YARD142,1476.86727762,727.23700124
YARD141,1507.25974003,727.77799722
YARD140,1541.05407716,728.37941078
YARD139,1483.19485498,727.36115290
YARD138,1512.57499338,727.88443000
YARD137,1543.23082798,728.43054232
YARD136,1489.48219953,727.48306566
YARD135,1516.38929005,727.96272422
YARD134,1545.11513752,728.47480481
YARD133,1494.32387883,727.57814871
YARD132,1519.73823328,728.03146636
YARD131,1475.86388113,727.25641152
YARD130,1498.62447805,727.66260579
YARD129,1523.44339967,728.10545671
YARD128,1480.78280217,727.35090893
YARD089,1027.01078485,719.29785919
YARD088,1027.00742133,719.29779865
YARD087,1027.00405162,719.29773800
YARD086,1027.00067728,719.29767726
YARD085,1026.99729672,719.29761641
YARD084,1026.99391072,719.29755546
YARD083,1026.99052004,719.29749443
YARD082,1026.98712192,719.29743327
YARD081,1026.98372045,719.29737204
YARD080,1026.98031270,719.29731070
YARD079,1026.97690023,719.29724928
YARD078,1026.97348145,719.29718774
YARD077,1026.97005713,719.29712610
YARD076,1026.96662805,719.29706438
YARD075,1026.96319257,719.29700254
YARD074,1026.95975251,719.29694062
YARD073,1026.95630608,719.29687859
YARD072,1026.95285405,719.29681645
YARD071,1026.94939718,719.29675423
YARD070,1026.94593390,719.29669189
YARD069,1026.94246575,719.29662946
YARD068,1026.93899229,719.29656694
YARD067,1026.93551220,719.29650430
YARD066,1026.93202720,719.29644157
YARD065,1026.92853573,719.29637873
YARD127,1027.13990828,719.30018171
YARD126,1027.13649875,719.30012037
YARD125,1027.13308771,719.30005900
YARD124,1027.12967686,719.29999763
YARD123,1027.12626449,719.29993624
YARD122,1027.12285232,719.29987485
YARD121,1027.11943863,719.29981343
YARD120,1027.11601773,719.29975189
YARD119,1027.11260374,719.29969046
YARD118,1027.10918823,719.29962901
YARD117,1027.10577291,719.29956757
YARD116,1027.10235609,719.29950609
YARD115,1027.09893945,719.29944462
YARD114,1027.09552130,719.29938313
YARD113,1027.08912043,719.29926857
YARD112,1027.08579610,719.29920875
YARD111,1027.08246377,719.29914879
YARD110,1027.07913587,719.29908890
YARD109,1027.07580486,719.29902896
YARD108,1027.07246633,719.29896889
YARD107,1027.06913173,719.29890889
YARD106,1027.06579403,719.29884883
YARD105,1027.06244927,719.29878864
YARD104,1027.05910715,719.29872850
YARD103,1027.05576355,719.29866834
YARD102,1027.05241254,719.29860804
YARD101,1027.04906370,719.29854778
YARD100,1027.04570938,719.29848742
YARD099,1027.04235609,719.29842708
YARD098,1027.03900050,719.29836670
YARD097,1027.03563990,719.29830623
YARD096,1027.03227985,719.29824577
YARD095,1027.02891749,719.29818526
YARD094,1027.02554979,719.29812467
YARD093,1027.02218377,719.29806410
YARD092,1027.01881462,719.29800347
YARD091,1027.01544061,719.29794276
YARD064,1026.94584227,719.29671496
YARD063,1026.94721390,719.29673964
YARD062,1026.94858440,719.29676430
YARD061,1026.94995383,719.29678895
YARD060,1026.95132208,719.29681357
YARD059,1026.95268950,719.29683817
YARD058,1026.95405564,719.29686276
YARD057,1026.95542052,719.29688732
YARD056,1026.95678448,719.29691186
YARD055,1026.95814713,719.29693638
YARD054,1026.95950890,719.29696089
YARD053,1026.96086963,719.29698538
YARD052,1026.96222893,719.29700984
YARD051,1026.96358744,719.29703428
YARD050,1026.96494458,719.29705870
YARD049,1026.96630087,719.29708311
YARD048,1026.96765615,719.29710750
YARD047,1026.96900992,719.29713186
YARD046,1026.97036305,719.29715621
YARD045,1026.97171493,719.29718054
YARD044,1026.97306539,719.29720484
YARD043,1026.97441527,719.29722913
YARD042,1026.97576353,719.29725339
YARD041,1026.97711128,719.29727765
YARD040,1026.97845765,719.29730187
YARD039,1026.97980295,719.29732608
YARD038,1026.98114714,719.29735027
YARD015,1027.01175437,719.29790111
YARD016,1027.01046248,719.29787786
YARD017,1027.00916931,719.29785459
YARD018,1027.00787423,719.29783128
YARD019,1027.00657785,719.29780795
YARD020,1027.00527956,719.29778458
YARD021,1027.00397965,719.29776118
YARD022,1027.00267845,719.29773777
YARD023,1027.00137531,719.29771431
YARD024,1027.00007127,719.29769084
YARD025,1026.99876518,719.29766734
YARD026,1026.99745716,719.29764379
YARD027,1026.99614751,719.29762022
YARD028,1026.99483654,719.29759663
YARD029,1026.99352362,719.29757300
YARD030,1026.99220907,719.29754934
YARD031,1026.99089318,719.29752566
YARD032,1026.98957534,719.29750194
YARD033,1026.98825635,719.29747820
YARD034,1026.98693550,719.29745443
YARD035,1026.98561269,719.29743062
YARD036,1026.98428823,719.29740678
YARD037,1026.98296241,719.29738292
YARD001,1027.03019343,719.29823297
YARD002,1027.02892131,719.29821007
YARD003,1027.02764853,719.29818716
YARD004,1027.02637295,719.29816420
YARD005,1027.02509635,719.29814122
YARD006,1027.02381667,719.29811819
YARD007,1027.02253653,719.29809515
YARD008,1027.02125304,719.29807205
YARD009,1027.01996904,719.29804893
YARD010,1027.01868172,719.29802576
YARD011,1027.01739353,719.29800258
YARD012,1027.01610369,719.29797936
YARD013,1027.01481060,719.29795609
YARD014,1027.01351654,719.29793279"""
    
    expected_coords = parse_expected_coords(csv_data)
    
    # Load crops metadata with manual spot mappings
    base_path = Path("out/crops/test-video")
    
    # Manual mappings: track_id -> spot_name
    folder_mappings = {
        "def364c7-6b84-4158-bbec-0ebc4e3eb1d0_IMG_1409": {
            1: "DD042",  # track001
            2: "DD045",  # track002
            3: "DD046",  # track003
            4: "DD048"   # track004
        },
        "9e648190-3503-4512-b698-76775f706c9a_IMG_1410": {
            1: "YARD182",  # track001
            2: "YARD181",  # track002
            3: "YARD175"   # track003
        }
    }
    
    all_matches = []
    
    for folder, spot_mapping in folder_mappings.items():
        folder_path = base_path / folder
        crops = load_crops_metadata(str(folder_path))
        matches = match_crops_to_spots(crops, expected_coords, folder, spot_mapping)
        all_matches.extend(matches)
        
        print(f"\n{folder}:")
        for crop, spot, coords in matches:
            bbox = crop.get('bbox_original', crop.get('bbox', []))
            track_id = crop.get('track_id')
            print(f"  Track {track_id} ({crop['crop_filename']}) -> {spot}")
            print(f"    BBox: {bbox}")
            print(f"    Expected: {coords}")
    
    if not all_matches:
        print("\nNo matches found. Please verify the crops metadata and expected coordinates.")
        return
    
    # Prepare data for calibration
    bboxes = [m[0].get('bbox_original', m[0].get('bbox', [])) for m in all_matches]
    expected = [m[2] for m in all_matches]
    
    print(f"\n{'='*80}")
    print(f"Calibrating with {len(all_matches)} samples")
    print(f"{'='*80}")
    
    # Test standard methods
    calibration_result = calibrate_bbox_method(bboxes, expected)
    
    print(f"\nBest method: {calibration_result['best_method']}")
    print(f"Average error: {calibration_result['avg_error']:.2f}px")
    print(f"\nMethod statistics:")
    for method, stats in calibration_result['method_stats'].items():
        print(f"  {method}:")
        print(f"    Avg: {stats['avg']:.2f}px, Max: {stats['max']:.2f}px, Min: {stats['min']:.2f}px, Std: {stats['std']:.2f}px")
    
    # Test advanced methods
    print(f"\n{'='*80}")
    print("Testing advanced methods...")
    print(f"{'='*80}")
    
    advanced_methods = [
        "learned_features",
        "perspective_corrected",
        "dimension_based"
    ]
    
    advanced_errors = {}
    for method in advanced_methods:
        errors = []
        for bbox, expected_coord in zip(bboxes, expected):
            try:
                calc_coords = calculate_image_coords_advanced(bbox, method)
                error = np.sqrt(
                    (calc_coords[0] - expected_coord[0])**2 +
                    (calc_coords[1] - expected_coord[1])**2
                )
                errors.append(error)
            except Exception as e:
                errors.append(float('inf'))
        
        avg_error = np.mean(errors) if errors else float('inf')
        advanced_errors[method] = {
            'avg': avg_error,
            'max': max(errors) if errors else float('inf'),
            'min': min(errors) if errors else float('inf'),
            'errors': errors
        }
        print(f"\n{method}:")
        print(f"  Avg: {avg_error:.2f}px, Max: {max(errors):.2f}px, Min: {min(errors):.2f}px")
    
    # Test linear combination (least squares)
    print(f"\n{'='*80}")
    print("Testing linear combination (least squares)...")
    print(f"{'='*80}")
    
    linear_result = find_optimal_linear_combination(bboxes, expected)
    print(f"\nLinear model results:")
    print(f"  Total error: {linear_result['total_error']:.2f}px")
    print(f"  X error: {linear_result['error_x']:.2f}px")
    print(f"  Y error: {linear_result['error_y']:.2f}px")
    
    if linear_result['weights_x'] is not None:
        print(f"\n  X weights:")
        for feat, weight in zip(linear_result['features'], linear_result['weights_x']):
            print(f"    {feat}: {weight:.6f}")
        
        print(f"\n  Y weights:")
        for feat, weight in zip(linear_result['features'], linear_result['weights_y']):
            print(f"    {feat}: {weight:.6f}")
        
        # Test linear model on samples
        print(f"\n  Per-sample results with linear model:")
        for match, bbox, expected_coord in zip(all_matches, bboxes, expected):
            calc_coords = calculate_with_linear_model(
                bbox,
                linear_result['weights_x'],
                linear_result['weights_y'],
                linear_result['features']
            )
            error = np.sqrt(
                (calc_coords[0] - expected_coord[0])**2 +
                (calc_coords[1] - expected_coord[1])**2
            )
            print(f"    {match[0]['crop_filename']} ({match[1]}): error={error:.2f}px")
        
        # Save configuration file
        config_path = Path("config/bbox_to_image_coords.json")
        config_path.parent.mkdir(exist_ok=True)
        
        config = {
            "method": "linear_combination",
            "linear_model": {
                "weights_x": linear_result['weights_x'].tolist(),
                "weights_y": linear_result['weights_y'].tolist(),
                "features": linear_result['features']
            },
            "fallback_method": "adaptive_percentage",
            "description": f"Linear combination model learned from calibration data. Total error: {linear_result['total_error']:.2f}px, X error: {linear_result['error_x']:.2f}px, Y error: {linear_result['error_y']:.2f}px."
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Configuration saved to: {config_path}")
        print(f"{'='*80}")
    
    # Find optimal percentage offsets with different reference points
    print(f"\n{'='*80}")
    print("Finding optimal percentage offsets...")
    print(f"{'='*80}")
    
    # Test different reference points
    reference_points = ["right_edge", "left_edge", "center"]
    best_overall = None
    best_error = float('inf')
    
    for ref_point in reference_points:
        print(f"\nTesting reference point: {ref_point}")
        optimal = find_optimal_percentage_offsets(
            bboxes, 
            expected,
            x_range=(-0.5, 0.5),  # Allow negative offsets and positive
            y_range=(0.0, 1.0),    # From top to bottom
            step=0.01,
            reference_point=ref_point
        )
        
        print(f"  X offset: {optimal['optimal_x_pct']*100:.2f}% (from {ref_point})")
        print(f"  Y offset: {optimal['optimal_y_pct']*100:.2f}% (from top)")
        print(f"  Average error: {optimal['avg_error']:.2f}px")
        
        if optimal['avg_error'] < best_error:
            best_error = optimal['avg_error']
            best_overall = optimal
    
    optimal = best_overall
    print(f"\n{'='*80}")
    print(f"BEST OVERALL CONFIGURATION:")
    print(f"{'='*80}")
    
    print(f"\nOptimal offsets:")
    print(f"  Reference point: {optimal['optimal_reference']}")
    print(f"  X offset: {optimal['optimal_x_pct']*100:.2f}% of width (from {optimal['optimal_reference']})")
    print(f"  Y offset: {optimal['optimal_y_pct']*100:.2f}% of height (from top)")
    print(f"  Average error: {optimal['avg_error']:.2f}px")
    
    print(f"\nTop 10 configurations:")
    for i, result in enumerate(optimal['all_results'], 1):
        print(f"  {i}. X: {result['x_pct']*100:.2f}%, Y: {result['y_pct']*100:.2f}% - Error: {result['avg_error']:.2f}px")
    
    # Show per-sample results with best method
    print(f"\n{'='*80}")
    print("Per-sample results with optimal method:")
    print(f"{'='*80}")
    for match, bbox, expected_coord in zip(all_matches, bboxes, expected):
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = (x1 + x2) / 2.0
        
        # Calculate X based on optimal reference point
        ref_point = optimal['optimal_reference']
        if ref_point == "right_edge":
            calc_x = float(x2) + (bbox_width * optimal['optimal_x_pct'])
        elif ref_point == "left_edge":
            calc_x = float(x1) + (bbox_width * optimal['optimal_x_pct'])
        elif ref_point == "center":
            calc_x = center_x + (bbox_width * optimal['optimal_x_pct'])
        else:
            calc_x = float(x2) + (bbox_width * optimal['optimal_x_pct'])
        
        calc_y = y1 + (bbox_height * optimal['optimal_y_pct'])
        
        error = np.sqrt((calc_x - expected_coord[0])**2 + (calc_y - expected_coord[1])**2)
        
        print(f"\n{match[0]['crop_filename']} ({match[1]}):")
        print(f"  BBox: {bbox}")
        print(f"  Calculated: ({calc_x:.2f}, {calc_y:.2f})")
        print(f"  Expected: ({expected_coord[0]:.2f}, {expected_coord[1]:.2f})")
        print(f"  Error: {error:.2f}px")


if __name__ == "__main__":
    main()



