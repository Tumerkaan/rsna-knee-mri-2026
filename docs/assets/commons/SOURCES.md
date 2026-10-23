# External illustrative MRI assets

These images are not competition data or training inputs. The exact shipped bytes
are pinned in [MEDIA_MANIFEST.json](../../../MEDIA_MANIFEST.json). The code license
does not replace their media licenses. Original videos and raw volumes are not in
this package. The original source review is retained privately.

| Shipped files | Source / author | Sequence and source selection | Media license |
|---|---|---|---|
| axial_full.png, axial_sq.png, axial_sq_flip.png, axial_sq_zm1.png, axial_sq_zp1.png, axial_rgb.png | [Knee MRI pd tse fs tra 320.ogg](https://commons.wikimedia.org/wiki/File:Knee_MRI_pd_tse_fs_tra_320.ogg), Ptrump16 | Axial PD TSE FS; 1190 × 896; 24 frames at 10 fps; zero-based center 7, neighbors 6 and 8 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| coronal_sq.png, coronal_sq_flip.png | [Knee MRI t2 tse fs cor 320.ogg](https://commons.wikimedia.org/wiki/File:Knee_MRI_t2_tse_fs_cor_320.ogg), Ptrump16 | Coronal T2 TSE FS; 1190 × 896; 22 frames at 10 fps; zero-based 13 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| sag_other_sq.png | [Knee MRI T1 TSE Sagittal.jpg](https://commons.wikimedia.org/wiki/File:Knee_MRI_T1_TSE_Sagittal.jpg), Ptrump16 | Sagittal T1 TSE without stated fat suppression; original 902 × 895 still; frame index not applicable | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| cor_other_sq.png; ../crop/scan1_full.png, scan1_mm.png, scan1_px.png, scan2_full.png, scan2_mm.png, scan2_px.png | [KMAR-50K v6](https://doi.org/10.17632/xw7mrg7ntg.6), Yuting Wang and Feng Shi | Coronal T1 TSE reference slice; zero-based 12 of 22; native 640 × 640; TR 600 ms, TE 9.8 ms, no FS protocol designation | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Frames extracted from **Knee MRI pd tse fs tra 320.ogg**, CC BY-SA 4.0.
Frame extracted from **Knee MRI t2 tse fs cor 320.ogg**, CC BY-SA 4.0.
The axial neighbors had different pixel content in the original extraction.
Frame indices describe the video, not DICOM coordinates.

Site derivatives were prepared by Tümer Kaan Mutlu. Square versions are
cropped/resized; flip versions are mirrored; axial_rgb combines three slices
into color channels. The crop demonstration uses the KMAR slice, with
cropping/resizing and simulated 0.5 mm/pixel downsampling. Derivatives retain
their source licenses; they are not unchanged scanner pixels. Their source
images were reviewed for burned-in text and annotations in the asset task.

## KneeMRI slices in ../kneemri/

[Zenodo record 14789903](https://zenodo.org/records/14789903), Clinical Hospital
Centre Rijeka, Croatia, Štajduhar et al., 2017.
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Eight PNGs retain the previously approved native exports with a common volume
display window. This update does not transform them. They are not competition
data or training inputs. See ATTRIBUTION.md for the full attribution.
