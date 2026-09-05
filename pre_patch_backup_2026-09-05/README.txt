Untouched originals, taken 2026-09-05 immediately before patching.
(The Xavier system clock reads year 2000; the real date is 2026-09-05.)

REPLACED in ~/model/main -- restore these to undo:
  bev_processor.py      composed remap LUT
  image_processing.py   cached stitcher blend weights
  navigate.py           same LUT ported in, dead cv2.cuda.undistort branch
                        removed, checkpoint model_state unwrap added

NOT replaced -- unmodified on disk, kept only as restore points:
  nav_processing.py
  steering_node.py
  pure_pursuit_controller.py

Roll back one file:
  cp -p bev_processor.py ~/model/main/

Roll back everything that was patched:
  cp -p bev_processor.py image_processing.py navigate.py ~/model/main/

Verify this folder is intact:
  cd ~/model/main/pre_patch_backup_2026-09-05 && md5sum -c MD5SUMS.txt

Also added to ~/model/main by the same work (new files, nothing overwritten):
  bev_processor_fixed.py  image_processing_fixed.py  bev_processor_orig.py
  run_navigate.sh  run_navigate.py
  bench_bev.py  bench_stitch.py  bench_nav.py  diffdist.py  mkcompare.py  live_bev.py
