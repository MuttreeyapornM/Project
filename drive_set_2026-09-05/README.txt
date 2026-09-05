Drive set (from BFV Google Drive folder, rear = rear_fixed_intrinsic).
Deployed 2026-09-05 and REVERTED: it magnifies the side and rear views badly on
the current physical board layout. Re-apply with:
  cp -p param_settings.py ~/model/main/ && cp -p yaml/*.yaml ~/model/main/yaml/
