import os
import sys
import traceback
from dotenv import load_dotenv
from config.args import parse_args, post_process_args
from distill import main as distill_main
from generate import main as mcts_main
from eval import main as eval_main
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Unified main entry point for Distillation and MCTS pipelines.
    """
    load_dotenv()
    args = parse_args()
    args = post_process_args(args)

    print(f"[INFO] Starting execution with mode: {args.mode}")

    try:
        # --- MODE 1: Distillation ---
        if args.mode == "distill":
            print("[INFO] Running distillation pipeline...")
            distill_main(args)

        # --- MODE 2: MCTS ---
        elif args.mode == "mcts":
            print("[INFO] Running MCTS reasoning pipeline...")
            assert args.if_use_cards == False, "Please set if_use_cards as False for MCTS"
            mcts_main(args)

        # --- MODE 3: Distilled MCTS ---
        elif args.mode == "distilled_mcts":
            print("[INFO] Running distilled MCTS...")
            assert args.if_use_cards == True, "Please set if_use_cards as True for distilled MCTS"            
            mcts_main(args)

        elif args.mode == "eval":
            print("[INFO] Running eval pipeline...")
            assert args.if_use_cards == True, "Please set if_use_cards as True for eval"   
            user_question = """Analyze high-resolution satellite images to identify and classify potential anomalies in the area and identify geospatial objects related to forest disaster scenarios. Utilize the available data to detect unusual patterns, segment various objects, and categorize the area to aid in effective disaster management and response planning. The data required for this task includes: (1) a main input image representing a forest area located at '/data/satellite_images/forest_area_main_image.tif', (2) a high spatial resolution (HSR) satellite image containing the geospatial area of interest located at '/data/satellite_images/geospatial_area_hsr_image.tif',  (4) a main satellite image containing 10 specific spectral bands from Sentinel-2 located at '/data/satellite_images/sentinel2_spectral_bands.tif', and (5) a binary mask array specifying visible regions during the inference process located at '/data/masks/visible_regions_mask.npy'."""        

            eval_answer, eval_completion = eval_main(args, user_question)
            print("[INFO] Eval answer generated...")
            print(eval_answer)
        else:
            print(f"[ERROR] Unknown mode '{args.mode}'. Expected one of ['distill', 'mcts', 'distilled_mcts', 'eval'].")

    except Exception as e:
        print(f"[FATAL] Error occurred during execution: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()