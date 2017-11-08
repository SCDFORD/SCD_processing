from scdutils.smart_functions import ants_simple_reg, apply_ants_warp, apply_lin_transform_flirt, apply_reverse_ants_warp, threshold_image
from scdutils.misc_utils import nifti_writer
import scdutils.directoryWalkers as directoryWalkers
from scdutils.space_functions import apply_linear_registration, perform_linear_registration
from scdutils.spreadsheet_maker import parse_spreadsheet_string
from collections import OrderedDict
from scdutils.misc_utils import write_ordered_dict_to_csv, read_ordered_dict_from_csv
import nibabel
import numpy
from subprocess import call
import os
from scdutils.image_calculator import nifti_equation, nifti_equation_smart


#Using pcasl 1.5s
filename_additions = """
cbf_GM_normalized        = @pcasl_1p5s_space/<#.nii.gz>
cbf_0_to_25_mask         =                  /<#.nii.gz>
cbf_25_to_35_mask        =                  /<#.nii.gz>
cbf_35_to_45_mask        =                  /<#.nii.gz>
cbf_45_plus_mask         =                  /<#.nii.gz>
"""

spreadsheet_format = """
--cbf_GM_normalized,      s:t1_segmented_with_subcort
    *3: 'WM rCBF'
--cbf_GM_normalized,    m:cbf_0_to_25_mask,      s:t1_segmented_with_subcort
    *3: 'WM rCBF <25'
    *#3: 'Num of voxels< 25'
    *V3: 'Volume of voxels< 25'
--cbf_GM_normalized,    m:cbf_25_to_35_mask,      s:t1_segmented_with_subcort
    *3: 'WM rCBF 25_35'
    *#3: 'Num of voxels 25_35'
    *V3: 'Volume of voxels 25_35' 
--cbf_GM_normalized,    m:cbf_35_to_45_mask,      s:t1_segmented_with_subcort
    *3: 'WM rCBF 35_45'
    *#3: 'Num of voxels 35_45'
    *V3: 'Volume of voxels 35_45'
--pcasl_1p5s.cbf_map,    m:cbf_0_to_25_mask,      s:t1_segmented_with_subcort
    *3: 'WM CBF (1.5 pCASL s); <25'
--pcasl_1p5s.cbf_map,    m:cbf_25_to_35_mask,     s:t1_segmented_with_subcort
    *3: 'WM CBF (1.5 pCASL s); 25_35'
--pcasl_1p5s.cbf_map,    m:cbf_35_to_45_mask,   s:t1_segmented_with_subcort
    *3: 'WM CBF (1.5 pCASL s); 35_45'       
--recalculated_oef,      m:cbf_0_to_25_mask,   s:t1_segmented_with_subcort, m:oef_mask
    *3: 'WM OEF (1p5s pcasl); 0 to 25'  
    *#3: 'Number of voxels (0 to 25)'
--recalculated_oef,      m:cbf_25_to_35_mask,   s:t1_segmented_with_subcort, m:oef_mask
    *3: 'WM OEF (1p5s pcasl); 25 to 35'  
    *#3: 'Number of voxels (25 to 35)'
--recalculated_oef,      m:cbf_35_to_45_mask,   s:t1_segmented_with_subcort, m:oef_mask
    *3: 'WM OEF (1p5s pcasl); 35 to 45'  
    *#3: 'Number of voxels (35 to 45)'
"""

if __name__=="__main__":
    from scdutils.config import SCDConfig
    from scdutils.common_v3 import config_filename, FilenameConstructor, parse_filesystem_from_string 
    cfg = SCDConfig(config_filename)
    loopControl = directoryWalkers.walkAllSubjects
    spreadsheet_filename = 'watershed_WM_pcasl_CBF.csv'
    tree = parse_filesystem_from_string(filename_additions) 
    mol_list = parse_spreadsheet_string(spreadsheet_format) 
    everyone_data = OrderedDict()
    for subjDir in loopControl(cfg.basedir): 
        #if not ("ASCD" in subjDir or "SCMR" in subjDir):
            #continue
        names = FilenameConstructor(subjDir, cfg)
        print(names) 
        names.use_parsed_schema(tree)
        
        perform_linear_registration(names.gm_pv_filename, names.pcasl_1p5s.cbf_map, cost="corratio", dof=6, search=True)
        apply_linear_registration(names.gm_pv_filename, names.pcasl_1p5s.cbf_map, interp='nearestneighbour')
        try:
            with nifti_writer(names.cbf_GM_normalized.in_native_space, template = names.pcasl_1p5s.cbf_map.in_native_space) as f:
                gm_mask = nibabel.load(str(names.gm_pv_filename.in_space(names.pcasl_1p5s_space))).get_data()>0.75
                print(f.data[0, 0, 0])
                f.data[f.data<5] = 0
                gm_cbf = f.data[gm_mask].mean()
                #print("-------------------", gm_cbf)
                f.data = f.data/gm_cbf
            with nifti_writer(names.t1_segmented_with_subcort.in_space(names.pcasl_1p5s_space), template = names.t1_segmented_with_subcort.in_space(names.pcasl_1p5s_space)) as f:
                f.data = numpy.round(f.data)
        except FileNotFoundError:
            pass


        threshold_image(names.cbf_GM_normalized.in_native_space, names.cbf_0_to_25_mask.in_native_space, lower_thr = 0.01, upper_thr = 0.25, binarize=True)
        perform_linear_registration(names.cbf_GM_normalized, names.recalculated_oef, cost="corratio", dof=6, search=True)
        apply_linear_registration(names.cbf_0_to_25_mask, names.recalculated_oef, interp='nearestneighbour')
        threshold_image(names.cbf_GM_normalized.in_native_space, names.cbf_25_to_35_mask.in_native_space, lower_thr = 0.25, upper_thr = 0.35, binarize=True)
        apply_linear_registration(names.cbf_25_to_35_mask, names.recalculated_oef, interp='nearestneighbour')
        threshold_image(names.cbf_GM_normalized.in_native_space, names.cbf_35_to_45_mask.in_native_space, lower_thr = 0.35, upper_thr = 0.45, binarize=True)
        apply_linear_registration(names.cbf_35_to_45_mask, names.recalculated_oef, interp='nearestneighbour')
        threshold_image(names.cbf_GM_normalized.in_native_space, names.cbf_45_plus_mask.in_native_space, lower_thr = 0.45, binarize=True)
        apply_linear_registration(names.cbf_45_plus_mask, names.recalculated_oef, interp='nearestneighbour') 
            
           
        perform_linear_registration(names.pcasl_1p5s.cbf_map, names.t1_skullstripped, cost="corratio", dof=6, search=True)
        apply_linear_registration(names.t1_segmented, names.cbf_0_to_25_mask, interp='nearestneighbour')
        apply_linear_registration(names.t1_segmented, names.cbf_25_to_35_mask, interp='nearestneighbour')
        apply_linear_registration(names.t1_segmented, names.cbf_35_to_45_mask, interp='nearestneighbour')
        
        subject_data = mol_list.process_subject(names)
        
        everyone_data[names.subject_name] = subject_data


    write_ordered_dict_to_csv(everyone_data, spreadsheet_filename)
