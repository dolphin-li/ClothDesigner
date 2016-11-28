
/* svg_files.c - list of SVG files for nvpr_svg to support. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if __APPLE__
#include <GL/glew.h>
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

#include "countof.h"

#include "svg_files.hpp"

#ifdef NDEBUG
const static int verbose = 0;
#else
const static int verbose = 1;
#endif

// Strings starting with an exclamation point are put into the "Demos..." menu.
static const char *svg_files[] = {

"!svg/complex/tiger.svg",  // complex, 239 paths, WITH stroking
"!svg/complex/Celtic_round_dogs.svg",  // complex, but just 1 path, no sroking
"!svg/complex/Coat_of_Arms_of_American_Samoa.svg", // 953 paths, some stroking
"!svg/complex/Buonaparte.svg",  // complex, 151 paths, WITH stroking
"!svg/complex/cowboy.svg", // AGG
"!svg/characters/nb_faces_nib_01.svg",
"!svg/characters/nb_faces_nib_04.svg",
"!svg/complex/emeza_Ocho.svg",
"!svg/complex/emeza_Vangogh.svg",

"!svg/complex/johnny_automatic_marching_band_1.svg",
"!svg/complex/american eagle.svg",
"!svg/complex/Cougar_.svg", // http://www.openclipart.org/people/papapishu/Cougar_.svg
"!svg/complex/Wormian_bones.svg",
"!svg/complex/Digital_Camera.svg",
"!svg/complex/Darth_Gimp_Cordless_Phone.svg",
"!svg/complex/gallardo.svg",  // uses linear gradients
"!svg/complex/Crown_of_Italy.svg",
"!svg/complex/papapishu_Boys_running.svg",
"!svg/complex/papapishu_galleon.svg",
"!svg/basic/recycling_box_3d_a.j._as_01.svg",

"!svg/basic/Yokozawa_Hiragana.svg",
"!svg/basic/alphaber-numbers-tracing.svg",
"!svg/basic/buggi_intricate_letter_A.svg",

"!svg/complex/tiger.svg",  // complex, 239 paths, WITH stroking
"!svg/complex/tiger_clipped_by_heart.svg",
"!svg/complex/tiger_clipped_by_cowboy.svg",
"!svg/simple/clipping_venn_diagrams.svg",

"!svg/characters/clinton.svg",
"!svg/characters/mcseem2.svg",

"!svg/simple/This_is_crazy.svg",
"!svg/simple/dash_This_is_crazy.svg",

"!svg/simple/dashed_examples.svg",  // contains arc commands!

"!svg/basic/cake1_jarno_vasamaa_01.svg",  // dashing with round endcaps (wrong now), gradients too
"!svg/simple/cake_flower.svg",
"!svg/basic/aircraft_jarno_vasamaa_.svg",  // dashing for airplane windows

"!svg/ravg/Two_red_dice.svg",  // Dice
"!svg/ravg/gotas.svg",  // Drops (formerly Gotas)
"!svg/simple/exampleRadGradFP.svg",
"!svg/simple/radgrad01.svg",  // sRGB vs. linear color space

"!svg/pacman/girl.svg",
"!svg/pacman/geisha.svg",
"!svg/pacman/candy.svg",
"!svg/pacman/pirate.svg",

"!svg/rpg/city.svg",
"!svg/rpg/statue.svg",
"!svg/rpg/statue_outline.svg",
"!svg/rpg/city_outline.svg",
"!svg/rpg/village.svg",

"!svg/icons/applications-multimedia.svg",
"!svg/icons/audio-volume-high.svg",
"!svg/icons/audio-volume-muted.svg",
"!svg/icons/document-print-preview.svg",
"!svg/icons/internet-mail.svg",
"!svg/icons/edit-cut.svg",
"!svg/icons/preferences-desktop-keyboard-shortcuts.svg",
"!svg/icons/face-devilish.svg",
"!svg/icons/face-glasses.svg",
"!svg/icons/face-grin.svg",
"!svg/icons/face-monkey.svg",

"!svg/complex/Welsh_dragon.svg",

"svg/complex/watercut-edit.svg", // http://www.openclipart.org/detail/ocean-theme-papercut-by-last-dino
"svg/complex/map-berlin-brandenburg_04r.svg", // http://www.openclipart.org/people/Anonymous/map-berlin-brandenburg_04r.svg
"svg/complex/Flamingo.svg",  // http://www.lovevectorfree.com/2010/05/pink-flamingo-by-lvf/
"svg/complex/sample_2.svg",  // http://vectormagic.com/online/sample
"svg/complex/ih_vector_pattern.svg",  // http://vector.tutsplus.com/freebies/patterns/different-vector-pattern/
"svg/complex/olho_vetorial.svg", // http://wiki.softwarelivre.org/InkscapeBrasil/Imagem76
"svg/complex/Cougar_.svg", // http://www.openclipart.org/people/papapishu/Cougar_.svg
"svg/complex/Gerald_G_Motorcycle_Clipart.svg", // http://www.openclipart.org/people/Gerald_G/Gerald_G_Motorcycle_Clipart.svg
"svg/complex/rejon_Supergirl.svg", // http://www.openclipart.org/detail/63
"svg/complex/anchorage_single.svg", // http://plurib.us/1shot/2008/anchorage/anchorage_single.svg
"svg/complex/spring_tree_final.svg",  // http://plurib.us/1shot/2007/spring_tree/
"svg/complex/eleven_below_single.svg",  // http://plurib.us/1shot/2008/eleven_below/
"svg/complex/tiger.svg",
"svg/complex/tiger_clipped_by_heart.svg",
"svg/complex/tiger_clipped_by_cowboy.svg",
// COMPLEX: complex SVG files
"svg/complex/help-books-aj.svg_aj_ash_01.svg",
"svg/complex/Chrisdesign_LP_Guitar_with_flametopfinish.svg",
"svg/complex/johnny_automatic_marching_band_1.svg",
"svg/complex/auto-insurance.svg",
"svg/complex/american eagle.svg",
"svg/complex/industrial commercial credit.svg",
"svg/complex/Coat_of_arms_of_Bolivia.svg",
"svg/complex/johnny_automatic_skeleton.svg",
"svg/complex/EricOrtner_Brooklyn_Bridge.svg",
"svg/complex/us_capitol_building_cli_01.svg",
"svg/complex/Saint_George_and_dragon_drawing.svg",
"svg/complex/Wormian_bones.svg",
"svg/complex/Machovka_harddisk.svg",
"svg/complex/ship sailing beside statue of liberty.svg",

//"svg/complex/Chrisdesign_Photorealistic_Green_Apple.svg",  // uses patterns & filters
"svg/complex/Yellowtrafficlight.svg",
"svg/complex/Digital_Camera.svg",
"svg/complex/Bulawayo_Zimbabwe_COA.svg",
"svg/complex/strawberry.svg",
"svg/complex/Saxophone.svg",
"svg/complex/marbles.svg",
"svg/complex/Chrisdesign_Golden_mask_Tutanchamun.svg",
"svg/complex/Darth_Gimp_Cordless_Phone.svg",
"svg/complex/girl-scout-rowing-boat.svg",
"svg/complex/papapishu_Boys_running.svg",
"svg/complex/johnny_automatic_ocean_liner.svg",
"svg/complex/papapishu_galleon.svg",
"svg/complex/beakman_Blue_eye.svg",
"svg/complex/Anonymous_USB_Type_A_dual_receptacle_1.svg",
"svg/complex/Anonymous_Chef.svg",
"svg/complex/Anonymous_Map_of_Europe.svg",
"svg/complex/Anonymous_Martin_Luther_King_Jr_03.svg",
"svg/complex/Bell214STDrawing.svg",
"svg/complex/Buonaparte.svg",
"svg/complex/Burmapeacockforhistory.svg",
"svg/complex/Ccross.svg",  // contains arc commands!
"svg/complex/Celtic_round_dogs.svg",
"svg/complex/Coat_of_Arms_of_American_Samoa.svg",
"svg/complex/Coat_of_arms_of_Bavaria.svg",
"svg/complex/Coat_of_arms_of_Finland.svg",
"svg/complex/Crown_of_Italy.svg",
"svg/complex/FoodChain.svg",  // has middle gradient
"svg/complex/Human.svg",
"svg/complex/Judah_Lion.svg",
"svg/complex/PSNJORD.svg",
"svg/complex/Tyranosaurus_rex_1.svg",
"svg/complex/Welsh_dragon.svg",
"svg/complex/architetto_architetto_fr_01.svg",
// Anti-Grain Graphics SVG examples: http://www.antigrain.com/svg/index.html
"svg/complex/b8.svg",  // AGG
"svg/complex/beach_trip_ganson.svg",
"svg/complex/busy_mom_with_child_and_02.svg",
"svg/complex/car_ride_ganson.svg",
"svg/complex/celtic.svg",
"svg/complex/cowboy.svg", // AGG
"svg/complex/elefantone.svg",
"svg/complex/emeza_Cangrejo.svg",
"svg/complex/emeza_Jazz3.svg",
"svg/complex/emeza_Ocho.svg",
"svg/complex/emeza_Vangogh.svg",
"svg/complex/gallardo.svg",  // uses linear gradients
"svg/complex/hikinh_ganson.svg",
"svg/complex/johnny_automatic_a_couple.svg",
"svg/complex/longhorn.svg",  // AGG
"svg/complex/mairin_Students.svg",
"svg/complex/my_lovely_baby_enrique_m_01.svg",
"svg/complex/orru_earth.svg",
"svg/complex/pace_e_bene__architetto__01.svg",
"svg/complex/picasso.svg",
"svg/complex/rich_young_man_counting_01.svg",
"svg/complex/sport_architetto_frances_01.svg",
"svg/complex/tiger.svg",
"svg/complex/tiger_clipped_by_heart.svg",
"svg/complex/war_drum_enrique_meza_c_01.svg",

"svg/basic/bookstack.svg", // http://www.openclipart.org/people/J_Alves/bookstack.svg
"svg/basic/television_comic_2.svg", // http://www.openclipart.org/people/rg1024/television_comic_2.svg
"svg/basic/beach_ball_01.svg", // http://www.openclipart.org/people/rg1024/beach_ball_01.svg
"svg/basic/spring.svg", // http://www.openclipart.org/people/rg1024/spring.svg
"svg/basic/rg1024_earth_globe.svg", // http://www.openclipart.org/people/rg1024/rg1024_earth_globe.svg
"svg/basic/spring_bicicleta.svg", // http://www.openclipart.org/people/rg1024/spring_bicicleta.svg

// BASIC: more than test case, less than complex scenes
"svg/basic/Farmeral_hops_-_illustrated.svg",
"svg/basic/egonpin_Caldero.svg",
"svg/basic/johnny_automatic_boy_playing_with_toy_truck.svg",
"svg/basic/recycling_box_3d_a.j._as_01.svg",
"svg/basic/ronaldrhouston_sport_car.svg",
"svg/basic/stefanolmo_Homebrewing_Manometer.svg",
"svg/basic/tomas_arad_grappe.svg",

"svg/basic/Yokozawa_Hiragana.svg",
"svg/basic/menu_example4.svg",
"svg/basic/johnny_automatic_geisha_playing_shamisen.svg",
"svg/basic/thumbs_up_nathan_eady_01.svg",  // radial gradient
"svg/basic/hungry_pilgrim_gerald_g_01r.svg",
"svg/basic/alphaber-numbers-tracing.svg",
"svg/basic/ryanlerch_decorative_letter_T.svg",
"svg/basic/buggi_intricate_letter_A.svg",
"svg/basic/liftarn_Orc.svg",  // uses gradient
"svg/basic/man_head_mikhail_a.medve_.svg",
"svg/basic/picnic_01.svg",
"svg/basic/spaghetti_bw.svg",
"svg/basic/menu_example_.svg",
"svg/basic/telefono_email_frolland_01.svg",
"svg/basic/ipurush_Compass.svg",  // uses dashing
"svg/basic/johnny_automatic_happy_chef.svg",
"svg/basic/shokunin_Eiffle_tower_Paris.svg",
"svg/basic/husband-and-wife.svg",
"svg/basic/Anonymous_Diffraction_through_a_slit.svg",  // ugh, uses opacity property (but only at leafs, not groups)
"svg/basic/radio_wireless_tower_cor_.svg",
"svg/basic/paperbag2_juliane_krug.svg",  // dashing with round endcaps
"svg/basic/tangram_erwan_02.svg",
"svg/basic/cake1_jarno_vasamaa_01.svg",  // dashing with round endcaps (wrong now), gradients too
"svg/basic/sunwheel_vikingdread_01.svg",
"svg/basic/factory_gabrielle_nowick_.svg",  // gradients
"svg/basic/navigation_display_panel_01.svg",  // dashing with circle join issue
"svg/basic/aircraft_jarno_vasamaa_.svg",  // dashing for airplane windows
"svg/basic/Circle_arc.svg",
"svg/basic/Film_reel.svg",  // needs arcs
"svg/basic/Heckert_GNU_white.svg",
"svg/basic/Mushroom.svg",  // needs blending
"svg/basic/P_human_body.svg",  // needs linear gradient support
"svg/basic/Schooner.svg",
"svg/basic/Signorina_in_viola.svg",
"svg/basic/Sites_interstitiels_cubique_a_faces_centrees.svg",
"svg/basic/Thole_(PSF).svg",
"svg/basic/Vorschriftszeichen_17.svg",
"svg/basic/Web-browser.svg",
"svg/basic/apples.svg",
"svg/basic/birthday_cake.svg",
"svg/basic/butterfly.svg",
"svg/basic/coat_of_arms_of_anglica_01.svg",
"svg/basic/contour_camel.svg",
"svg/basic/emeza_Dolphin.svg",
"svg/basic/emeza_Guacamaya.svg",
"svg/basic/guitar_ganson.svg",
"svg/basic/knot_bowline.svg",
"svg/basic/lion.svg",
"svg/basic/mapsym.svg",
"svg/basic/paths-data-03-f.svg",
"svg/basic/silouettedonnavestiti.svg",
"svg/basic/symbols.svg",
"svg/basic/umbrella.svg",
"svg/basic/white_s_10.svg",
"svg/basic/white_s_q.svg",
"svg/basic/human_heart.svg",

// CHARACTERS: SVG content portraying people
"svg/characters/beethoveo_ganson.svg",
"svg/characters/blueman_306_01.svg",
"svg/characters/clinton.svg",
"svg/characters/comic-char-blonde.svg",
"svg/characters/comic-char-fighter.svg",
"svg/characters/comic-char-general.svg",
"svg/characters/comic-char-graduated.svg",
"svg/characters/comic-char-lady.svg",
"svg/characters/comic-char-pirate.svg",
"svg/characters/comic-char-santa.svg",
"svg/characters/einstein_01.svg",
"svg/characters/enthusiast.svg",
"svg/characters/mcseem2.svg",
"svg/characters/nb_faces_iod_01.svg",
"svg/characters/nb_faces_iod_02.svg",
"svg/characters/nb_faces_nib_01.svg",
"svg/characters/nb_faces_nib_04.svg",
"svg/characters/nb_faces_oac_01.svg",
"svg/characters/woman_chemist_gerald_g._01.svg",
"svg/characters/woman_doctor_gerald_g._01.svg",
"svg/characters/woman_police_01_gerald_g_01.svg",
"svg/characters/woman_reading_gerald_g._01.svg",
"svg/characters/xenia4.svg",

"svg/pacman/girl.svg",
"svg/pacman/geisha.svg",
"svg/pacman/candy.svg",
"svg/pacman/pirate.svg",
"svg/pacman/dandy.svg",

"svg/rpg/city.svg",
"svg/rpg/city_outline.svg",
"svg/rpg/statue.svg",
"svg/rpg/statue_outline.svg",
"svg/rpg/village.svg",
"svg/rpg/village_outline.svg",

// ESSENTIALS: examples from http://oreilly.com/catalog/svgess/chapter/ch03.html
"svg/essentials/basic-lines.svg",
"svg/essentials/circles-and-ellipses.svg",
"svg/essentials/fill-rules.svg",
"svg/essentials/polygon-element.svg",
"svg/essentials/polyline-element.svg",
"svg/essentials/rectangle-element.svg",
"svg/essentials/rounded-rectangles.svg",
"svg/essentials/stroke-color.svg",
"svg/essentials/stroke-dasharray.svg",
"svg/essentials/stroke-opacity.svg",
"svg/essentials/stroke-width.svg",

// FLAGS: typically complex geometry
"svg/flags/andorre_flag_patricia_fi_01.svg",
"svg/flags/australia_mike_honeychur_01.svg",
"svg/flags/brazil_flag_rob_lucas_01.svg",
"svg/flags/british_flag_felipescu_01r.svg",
"svg/flags/canada_flag_ganson.svg",
"svg/flags/cymru_flag_wales_michae_.svg",
"svg/flags/equatorial_guinea.svg",
"svg/flags/eritrea.svg",
"svg/flags/florence_flag_loz_.svg",
"svg/flags/kansasflag_dave_reckonin_01.svg",
"svg/flags/northern_mariana.svg",
"svg/flags/paraguay.svg",
"svg/flags/saint_helena.svg",
"svg/flags/south_georgia_and_south_sandwich_islands.svg",
"svg/flags/st_pierre_miquelon_patri_01.svg",
"svg/flags/tibet.svg",
"svg/flags/usa_rhode_island.svg",

// GHOSTSCRIPT: SVG files converted from PostScript
// Converted from Encapsulated PostScript to SVG with
// gswin32.exe -sDEVICE=svg -dSAFER -sOutputFile=gs_$i:r.svg -dNOPAUSE -dBATCH $i
"svg/ghostscript/gs_chess.svg",  // gobs of rects
//"svg/ghostscript/gs_alphabet.svg",  // gobs and gobs of rects
"svg/ghostscript/gs_colorcir.svg",  // gobs of rects
"svg/ghostscript/gs_doretree.svg",
"svg/ghostscript/gs_escher.svg",
"svg/ghostscript/gs_golfer.svg",
"svg/ghostscript/gs_grayalph.svg",  // gobs of rects
"svg/ghostscript/gs_ridt91.svg",  // gobs of rects for gradients!
"svg/ghostscript/gs_snowflak.svg",
"svg/ghostscript/gs_tiger.svg",
"svg/ghostscript/gs_vasarely.svg",
//"svg/ghostscript/gs_waterfal.svg",  // terrible scene: text decomposed into gobs of rects
"svg/ghostscript/linux_graph.svg",

// MISC: random stuff
"svg/misc/Tacchino_di_Natale.svg",
"svg/misc/lampada_da_cantiere.svg",
"svg/misc/landscape_near_the_river_01.svg",
"svg/misc/provette_architetto_fran_01.svg",
"svg/misc/what_have_you_done_dani_.svg",

// SIMPLE: category for simple test cases
"svg/simple/molumen_Guilloche.svg", // http://www.openclipart.org/people/molumen/molumen_Guilloche.svg
"svg/simple/jenkov_basic_shapes.svg",
"svg/simple/jenkov_gradient_linear_1.svg",
"svg/simple/jenkov_gradient_radial_1.svg",
"svg/simple/jenkov_gradients.svg",
"svg/simple/jenkov_images.svg",
"svg/simple/jenkov_layering.svg",
"svg/simple/jenkov_svg_element.svg",
"svg/simple/jenkov_transformations.svg",

"svg/simple/image.svg",  // uses <image> to show ghost.png
"svg/simple/pi.svg",
"svg/simple/Theta.svg",
"svg/simple/AtSign.svg",

"svg/simple/simple.svg",  // https://developer.mozilla.org/en/Mozilla_SVG_Project
// from http://apike.ca/prog_svg_clip.html
"svg/simple/exampleClip.svg",
"svg/simple/exampleClip2.svg",

// from http://www.carto.net/papers/svg/samples/stroking.shtml
"svg/simple/stroking_complex.svg",
"svg/simple/stroking_dashoffset.svg",
"svg/simple/stroking_linecap.svg",
"svg/simple/stroking_linejoin.svg",
"svg/simple/stroking_miterlimit.svg",
"svg/simple/stroking_strokewidth.svg",
// http://www.carto.net/papers/svg/samples/matrix.shtml
"svg/simple/matrix.svg",
// http://www.carto.net/papers/svg/samples/shapes.shtml
"svg/simple/shapes.svg",
// http://www.carto.net/papers/svg/samples/polys.shtml
"svg/simple/polys.svg",

"svg/simple/gotas_subset.svg",
"svg/simple/opacity_overlaps.svg",
"svg/simple/dash.svg",
"svg/simple/camera_linear_gradient_case.svg",
"svg/simple/lingrad01.svg",
"svg/simple/lingrad02.svg",
"svg/simple/lingrad03.svg",
"svg/simple/lingrad04.svg",
"svg/simple/lingrad05.svg",
"svg/simple/lingrad06.svg",
"svg/simple/lingrad07.svg",
"svg/simple/radgrad01.svg",
"svg/simple/radgrad02.svg",
"svg/simple/user_space_radial_gradient.svg",  // userSpaceOnUse radial gradient
"svg/simple/clipping_venn_diagrams.svg",
"svg/simple/clipping_intersection_test_1.svg",
"svg/simple/clipping_intersection_test_2.svg",
"svg/simple/clip-merge_modes.svg",
"svg/simple/arc_bounds.svg", // regression test for getBounds on arcs

"svg/simple/radial1.svg",
"svg/simple/radial2.svg",

"svg/simple/Ambox_grammar.svg",
"svg/simple/bubbles.svg",

"svg/simple/exampleGradMethod.svg",
"svg/simple/exampleGradStops.svg",
"svg/simple/exampleRadGradFP.svg",

"svg/simple/hard_cusp_case.svg",
"svg/simple/cake_flower.svg",
"svg/simple/Ccross_arc.svg",  // contains arc commands!
"svg/simple/ContinuBJK.svg",
"svg/simple/Mushroom_fragment.svg",  // needs blending
"svg/simple/bad.svg",
"svg/simple/This_is_crazy.svg",
"svg/simple/dash_This_is_crazy.svg",
"svg/simple/arc.svg",
"svg/simple/arc2.svg",
"svg/simple/arc3.svg",
"svg/simple/arcs01.svg",
"svg/simple/arcs01_just_arc.svg",
"svg/simple/brain_jon_phillips_01.svg",
"svg/simple/circle01.svg",
"svg/simple/clinton_forehead.svg",
"svg/simple/cubic_serpentine.svg",
"svg/simple/cubic_serpentine_round.svg",
"svg/simple/cubic_serpentine_square.svg",
"svg/simple/cubic_serpentine_triangle.svg",
"svg/simple/cubic_stroke.svg",
"svg/simple/cubic_stroke_miter_truncate.svg",
"svg/simple/cubic_stroke_round.svg",
"svg/simple/dashed_examples.svg",  // contains arc commands!
"svg/simple/dashed_arcs.svg",  // contains arc commands!
"svg/simple/dashed_arcs2.svg",  // contains arc commands!
"svg/simple/dashed_arcs3.svg",  // contains arc commands!
"svg/simple/dashed_cubic_serpentine.svg",
"svg/simple/dashed_cubic_serpentine2.svg",
"svg/simple/dashed_moveto.svg",
"svg/simple/dashed_cubic_cusp.svg",
"svg/simple/dashed_lines.svg",
"svg/simple/dashed_lines2.svg",
"svg/simple/dashed_quadratic_stroke.svg",
"svg/simple/dashed_rotated_ellipse.svg",
// lots of cubic commands where the start and end-point are identical and extrapolating control points are nearly co-linear!
"svg/simple/degenerate_cubic_loops.svg",
"svg/simple/degenerate_cubic_loops2.svg",
"svg/simple/degenerate_dashed_cubic_loops.svg",
"svg/simple/ellipse01.svg",
"svg/simple/frame.svg",
"svg/simple/ghost_image.svg",
"svg/simple/ghost_svg_image.svg",
"svg/simple/line01.svg",
"svg/simple/line_stroke.svg",
"svg/simple/line_stroke2.svg",
"svg/simple/line_to_cubic.svg",
"svg/simple/linecap.svg",
"svg/simple/musawir_path.svg",
"svg/simple/normal_ellipse.svg",
"svg/simple/paths.svg",
"svg/simple/phone2.svg",
"svg/simple/dashed_phone2.svg",
"svg/simple/polygon01.svg",
"svg/simple/polyline01.svg",
"svg/simple/quadratic_stroke.svg",
"svg/simple/quadratic_stroke2.svg",
"svg/simple/quadratic_stroke_single.svg",
"svg/simple/rect01.svg",
"svg/simple/rect01a.svg",
"svg/simple/rect02.svg",
"svg/simple/rect02a.svg",
"svg/simple/rotated_ellipse.svg",
"svg/simple/rounded_box-isolated.svg",
"svg/simple/skew.svg",
"svg/simple/skew1.svg",
"svg/simple/spikes.svg",
"svg/simple/stroke_whisker.svg",
"svg/simple/weird_arc.svg",
"svg/simple/whisker.svg",
"svg/simple/whisker2.svg",
"svg/simple/whisker3.svg",
"svg/simple/whisker4.svg",
"svg/simple/whisker5.svg",
"svg/simple/zoom_case.svg",

// TEST cases from:
"svg/test/masking-intro-01-f.svg",
"svg/test/masking-mask-01-b.svg",
"svg/test/masking-opacity-01-b.svg",
"svg/test/masking-path-01-b.svg",
"svg/test/masking-path-02-b.svg",
"svg/test/masking-path-03-b.svg",
"svg/test/masking-path-04-b.svg",
"svg/test/masking-path-05-f.svg",
"svg/test/masking-path-06-b.svg",

"svg/test/Use01.svg",
"svg/test/Use01-GeneratedContent.svg",
"svg/test/Use02.svg",
"svg/test/Use02-GeneratedContent.svg",
"svg/test/Use03.svg",
"svg/test/Use03-GeneratedContent.svg",
"svg/test/Use04.svg",
"svg/test/Use04-GeneratedContent.svg",

// http://srufaculty.sru.edu/david.dailey/svg/newstuff/rainbow.svg
// "svg/test/gradient11c.svg",  exhibits high frequency gradients and <animate>
"svg/test/gradient10.svg",
"svg/test/ellipses2.svg",
"svg/test/use4.svg",
"svg/test/circles3.svg",
"svg/test/rainbow.svg",
"svg/test/ovaling.svg",
// http://www.w3.org/Graphics/SVG/Test/20061213/htmlObjectHarness/tiny-index.html
// http://dev.w3.org/SVG/profiles/1.1F2/test/svg/
// http://www.w3.org/Graphics/SVG/Test/20080912/W3C_SVG_12_TinyTestSuite.tar.gz
"svg/test/color-prop-03-t.svg",
"svg/test/coords-coord-01-t.svg",
"svg/test/coords-coord-02-t.svg",
"svg/test/coords-trans-02-t.svg",
"svg/test/coords-trans-03-t.svg",
"svg/test/coords-trans-04-t.svg",
"svg/test/coords-trans-05-t.svg",
"svg/test/coords-trans-06-t.svg",
"svg/test/painting-fill-01-t.svg",
"svg/test/painting-fill-02-t.svg",
"svg/test/painting-fill-03-t.svg",
"svg/test/painting-fill-04-t.svg",
"svg/test/painting-fill-05-b.svg",
"svg/test/painting-stroke-01-t.svg",
"svg/test/painting-stroke-02-t.svg",
"svg/test/painting-stroke-03-t.svg",
"svg/test/painting-stroke-04-t.svg",
"svg/test/painting-stroke-05-t.svg",
"svg/test/painting-stroke-06-t.svg",
"svg/test/painting-stroke-07-t.svg",
"svg/test/painting-stroke-08-t.svg",
"svg/test/painting-stroke-09-t.svg",
"svg/test/painting-stroke-10-t.svg",
"svg/test/paint-stroke-202-t.svg",
"svg/test/paint-color-201-t.svg",
"svg/test/paths-data-01-t.svg",
"svg/test/paths-data-02-t.svg",
"svg/test/paths-data-04-t.svg",
"svg/test/paths-data-05-t.svg",
"svg/test/paths-data-06-t.svg",
"svg/test/paths-data-07-t.svg",
"svg/test/paths-data-08-t.svg",
"svg/test/paths-data-09-t.svg",
"svg/test/paths-data-10-t.svg",
"svg/test/paths-data-12-t.svg",
"svg/test/paths-data-13-t.svg",
"svg/test/paths-data-14-t.svg",
"svg/test/paths-data-15-t.svg",
"svg/test/paths-data-16-t.svg",

"svg/test/pservers-grad-01-b.svg",
"svg/test/pservers-grad-02-b.svg",
"svg/test/pservers-grad-03-b.svg",  // pattern
"svg/test/pservers-grad-04-b.svg",
"svg/test/pservers-grad-05-b.svg",
"svg/test/pservers-grad-06-b.svg",
"svg/test/pservers-grad-07-b.svg",
"svg/test/pservers-grad-08-b.svg",  // gradients on text
"svg/test/pservers-grad-09-b.svg",
"svg/test/pservers-grad-10-b.svg",
"svg/test/pservers-grad-11-b.svg",
"svg/test/pservers-grad-12-b.svg",
"svg/test/pservers-grad-13-b.svg",  // ???
"svg/test/pservers-grad-14-b.svg",
"svg/test/pservers-grad-15-b.svg",
"svg/test/pservers-grad-16-b.svg",
"svg/test/pservers-grad-17-b.svg",
"svg/test/pservers-grad-18-b.svg",
"svg/test/pservers-grad-18-b-old.svg",
"svg/test/pservers-grad-19-b.svg",
"svg/test/pservers-grad-20-b.svg",
"svg/test/pservers-grad-21-b.svg",
"svg/test/pservers-grad-22-b.svg",

"svg/test/shapes-circle-01-t.svg",
"svg/test/shapes-circle-02-t.svg",
"svg/test/shapes-circle-03-t.svg",
"svg/test/shapes-ellipse-01-t.svg",
"svg/test/shapes-ellipse-02-t.svg",
"svg/test/shapes-ellipse-03-t.svg",
"svg/test/shapes-intro-01-t.svg",
"svg/test/shapes-line-01-t.svg",
"svg/test/shapes-polygon-01-t.svg",
"svg/test/shapes-polyline-01-t.svg",
"svg/test/shapes-rect-01-t.svg",
"svg/test/shapes-rect-02-t.svg",
"svg/test/shapes-rect-03-t.svg",
"svg/test/shapes-rect-03-t_alt.svg",
"svg/test/struct-group-01-t.svg",
"svg/test/struct-group-02-b.svg",
"svg/test/struct-group-03-t.svg",

// Various "hard" bezier test cases
"svg/bezier/fill_glitch.svg",  //WRONG
"svg/bezier/almost_outside_cubic_line_--_WRONG.svg",  //WRONG
"svg/bezier/exact_cusp.svg", // WRONG. cusp missing
"svg/bezier/kink_--_WRONG.svg",  // WRONG, notched in middle at bottom
"svg/bezier/nearly_cusp_--_WRONG.svg",  // GL looks bettre than OpenVG RI
"svg/bezier/outside_quadratic_line_--_WRONG.svg",  // WRONG
"svg/bezier/line_case.svg",  // WRONG
"svg/bezier/quadratic_thats_almost_a_circle.svg",  // OpenVG shows no crack in circle, hard to say who's right
"svg/bezier/tricky_bend,_almost_180_degree_end-point_turn.svg",  // GL has notch, but OpenVG doesn't
"svg/bezier/tricky_bend_assertion.svg",  // WRONG in filling, speckled pixels on one hull of loop

"svg/bezier/2nd.svg",
"svg/bezier/almost_cubic_triple_X-X-Y-X.svg",
"svg/bezier/almost_outside_cubic_line.svg",
"svg/bezier/almost_outside_quadratic_line.svg",
"svg/bezier/alternative_lazy_loop.svg",
"svg/bezier/another_huge_quadratic_hull.svg",
"svg/bezier/centerion_head.svg",
"svg/bezier/cubic_loop_with_hole,_reverse_order.svg",
"svg/bezier/cubic_loop_with_hole.svg",
"svg/bezier/cubic_loop_with_hole_and_negative_radius.svg",
"svg/bezier/cubic_loop_without_hole.svg",
"svg/bezier/cubic_serpentine.svg",
"svg/bezier/cubic_triple_X-X-Y-X_--_WRONG.svg",
"svg/bezier/cusp_with_empty_hull_--_WRONG.svg",
"svg/bezier/easy_wishbone_quadratic.svg",
"svg/bezier/hard_hole.svg",
"svg/bezier/hard_quadratic_extension_case.svg",
"svg/bezier/huge_quadratic_hull_with_3_nearby_ctrl_points_+_big_radius.svg",
"svg/bezier/huge_quadratic_with_knotched_edge,_far_off_control_point,_almost_deeply_recursive_cusp.svg",
"svg/bezier/huge_quadratic_with_knotched_edge,_far_off_control_point,_deeply_recursive_cusp.svg",
"svg/bezier/huge_radius_quadratic.svg",
"svg/bezier/huge_radius_quadratic_hull_escape.svg",
"svg/bezier/huge_radius_quadratic_with_near_mid-vertex.svg",
"svg/bezier/inside_cubic_line.svg",
"svg/bezier/inside_cubic_line_--_WRONG.svg",
"svg/bezier/inside_quadratic_line.svg",
"svg/bezier/lazy_loop.svg",
"svg/bezier/narrow_quadratic_with_far_far_extrapolating_control_point,_transitioned_to_hull.svg",
"svg/bezier/narrow_quadratic_with_far_far_extrapolating_control_point.svg",
"svg/bezier/outside_cubic_line_--_WRONG.svg",
"svg/bezier/quadratic_line.svg",
"svg/bezier/quadratic_over-bend.svg",
"svg/bezier/quadratic_over-bend_with_negative_radius.svg",
"svg/bezier/quadratic_with_far_far_extrapolating_control_point_(hull_sensitive).svg",
"svg/bezier/ropey_nearly_inside_quadratic_line.svg",
"svg/bezier/short_quadratic.svg",
"svg/bezier/small_nick.svg",
"svg/bezier/sneaky_lazy_loop.svg",
"svg/bezier/tight_narrow_serpentine.svg",
"svg/bezier/tight_turn_with_inflection.svg",
"svg/bezier/toes_to_fingers_loop.svg",
"svg/bezier/triple_point_cubic_X-X-X-Y.svg",
"svg/bezier/very_tight_narrow_serpentine.svg",
"svg/bezier/wide_cubic_serpentine,_almost_wrong.svg",
"svg/bezier/wide_cubic_serpentine_--_WRONG.svg",

"svg/ms/45CAD_Floor_Plan.htm",
"svg/ms/46InteractiveWorldMap.htm",
"svg/ms/areachart.htm",
"svg/ms/piechart.htm",

// from RAVG's ACKNOWLEDGE.txt
"svg/ravg/butterfly.svg",  // Butterfly
"svg/ravg/Breakdance_juliane_krug_01.svg",  // Dancer (formerly breakdance)
// Desktop, see svg/icons
"svg/ravg/Two_red_dice.svg",  // Dice
"svg/ravg/Anonymous_blue_dragonfly.svg",  // Dragonfly
"svg/ravg/gotas.svg",  // Drops (formerly Gotas)
"svg/ravg/jonata_Embrace_the_World.svg",  // Embrace (formerly Reciclagem)
"svg/ravg/Anonymous_Eyes.svg",  // Eyes
"svg/ravg/Anonymous_Glass_2.svg", // Glass
"svg/ravg/warszawianka_Hygieia.svg",  // Hygieia
"svg/ravg/lion.svg",  // Lion
// Penguin (formerly tux) is AWOL, couldn't find it
"svg/ravg/Anonymous_Scorpion.svg",  // Scorpion
"svg/ravg/Gerald_G_Roller_Blader.svg",  // Skater (formerly rollerblader)
"svg/ravg/tiger.svg",  // Tiger

// Desktop
// Selection of icons from the Discovery Icon Theme
// http://gnome-look.org/content/show.php?content=69950
"svg/icons/accessories-calculator.svg",
"svg/icons/accessories-character-map.svg",
"svg/icons/accessories-text-editor.svg",
"svg/icons/address-book-new.svg",
"svg/icons/alacarte.svg",
"svg/icons/application-certificate.svg",
"svg/icons/application-x-executable.svg",
"svg/icons/applications-accessories.svg",
"svg/icons/applications-development.svg",
"svg/icons/applications-games.svg",
"svg/icons/applications-graphics.svg",
"svg/icons/applications-internet.svg",
"svg/icons/applications-multimedia.svg",
"svg/icons/applications-office.svg",
"svg/icons/applications-other.svg",
"svg/icons/applications-system.svg",
"svg/icons/appointment-new.svg",
"svg/icons/audio-card.svg",
"svg/icons/audio-input-microphone.svg",
"svg/icons/audio-volume-high.svg",
"svg/icons/audio-volume-low.svg",
"svg/icons/audio-volume-medium.svg",
"svg/icons/audio-volume-muted.svg",
"svg/icons/audio-x-generic.svg",
"svg/icons/battery-caution.svg",
"svg/icons/battery.svg",
"svg/icons/bookmark-new.svg",
"svg/icons/camera-photo.svg",
"svg/icons/camera-video.svg",
"svg/icons/computer.svg",
"svg/icons/contact-new.svg",
"svg/icons/dialog-error.svg",
"svg/icons/dialog-information.svg",
"svg/icons/dialog-warning.svg",
"svg/icons/document-new.svg",
"svg/icons/document-open.svg",
"svg/icons/document-print-preview.svg",
"svg/icons/document-print.svg",
"svg/icons/document-properties.svg",
"svg/icons/document-save-as.svg",
"svg/icons/document-save.svg",
"svg/icons/drive-harddisk.svg",
"svg/icons/drive-optical.svg",
"svg/icons/drive-removable-media.svg",
"svg/icons/edit-clear.svg",
"svg/icons/edit-copy.svg",
"svg/icons/edit-cut.svg",
"svg/icons/edit-delete.svg",
"svg/icons/edit-find-replace.svg",
"svg/icons/edit-find.svg",
"svg/icons/edit-paste.svg",
"svg/icons/edit-redo.svg",
"svg/icons/edit-select-all.svg",
"svg/icons/edit-undo.svg",
"svg/icons/emblem-favorite.svg",
"svg/icons/emblem-important.svg",
"svg/icons/emblem-photos.svg",
"svg/icons/emblem-readonly.svg",
"svg/icons/emblem-symbolic-link.svg",
"svg/icons/emblem-system.svg",
"svg/icons/emblem-unreadable.svg",
"svg/icons/face-angel.svg",
"svg/icons/face-crying.svg",
"svg/icons/face-devilish.svg",
"svg/icons/face-glasses.svg",
"svg/icons/face-grin.svg",
"svg/icons/face-kiss.svg",
"svg/icons/face-monkey.svg",
"svg/icons/face-plain.svg",
"svg/icons/face-sad.svg",
"svg/icons/face-smile-big.svg",
"svg/icons/face-smile.svg",
"svg/icons/face-surprise.svg",
"svg/icons/face-wink.svg",
"svg/icons/folder-drag-accept.svg",
"svg/icons/folder-new.svg",
"svg/icons/folder-open.svg",
"svg/icons/folder-remote-ftp.svg",
"svg/icons/folder-remote-smb.svg",
"svg/icons/folder-remote-ssh.svg",
"svg/icons/folder-remote.svg",
"svg/icons/folder-saved-search.svg",
"svg/icons/folder-visiting.svg",
"svg/icons/folder.svg",
"svg/icons/font-x-generic.svg",
"svg/icons/format-indent-less.svg",
"svg/icons/format-indent-more.svg",
"svg/icons/format-justify-center.svg",
"svg/icons/format-justify-fill.svg",
"svg/icons/format-justify-left.svg",
"svg/icons/format-justify-right.svg",
"svg/icons/format-text-bold.svg",
"svg/icons/format-text-italic.svg",
"svg/icons/format-text-strikethrough.svg",
"svg/icons/format-text-underline.svg",
"svg/icons/go-bottom.svg",
"svg/icons/go-down.svg",
"svg/icons/go-first.svg",
"svg/icons/go-home.svg",
"svg/icons/go-jump.svg",
"svg/icons/go-last.svg",
"svg/icons/go-next.svg",
"svg/icons/go-previous.svg",
"svg/icons/go-top.svg",
"svg/icons/go-up.svg",
"svg/icons/help-browser.svg",
"svg/icons/image-loading.svg",
"svg/icons/image-missing.svg",
"svg/icons/image-x-generic.svg",
"svg/icons/input-gaming.svg",
"svg/icons/input-keyboard.svg",
"svg/icons/input-mouse.svg",
"svg/icons/internet-group-chat.svg",
"svg/icons/internet-mail.svg",
"svg/icons/internet-news-reader.svg",
"svg/icons/internet-web-browser.svg",
"svg/icons/list-add.svg",
"svg/icons/list-remove.svg",
"svg/icons/mail-attachment.svg",
"svg/icons/mail-forward.svg",
"svg/icons/mail-mark-junk.svg",
"svg/icons/mail-message-new.svg",
"svg/icons/mail-reply-all.svg",
"svg/icons/mail-reply-sender.svg",
"svg/icons/mail-send-receive.svg",
"svg/icons/media-eject.svg",
"svg/icons/media-flash.svg",
"svg/icons/media-floppy.svg",
"svg/icons/media-optical.svg",
"svg/icons/media-playback-pause.svg",
"svg/icons/media-playback-start.svg",
"svg/icons/media-playback-stop.svg",
"svg/icons/media-record.svg",
"svg/icons/media-seek-backward.svg",
"svg/icons/media-seek-forward.svg",
"svg/icons/media-skip-backward.svg",
"svg/icons/media-skip-forward.svg",
"svg/icons/multimedia-player.svg",
"svg/icons/network-error.svg",
"svg/icons/network-idle.svg",
"svg/icons/network-offline.svg",
"svg/icons/network-receive.svg",
"svg/icons/network-server.svg",
"svg/icons/network-transmit-receive.svg",
"svg/icons/network-transmit.svg",
"svg/icons/network-wired.svg",
"svg/icons/network-wireless-encrypted.svg",
"svg/icons/network-wireless.svg",
"svg/icons/network-workgroup.svg",
"svg/icons/office-calendar.svg",
"svg/icons/package-x-generic.svg",
"svg/icons/preferences-desktop-accessibility.svg",
"svg/icons/preferences-desktop-assistive-technology.svg",
"svg/icons/preferences-desktop-font.svg",
"svg/icons/preferences-desktop-keyboard-shortcuts.svg",
"svg/icons/preferences-desktop-locale.svg",
"svg/icons/preferences-desktop-peripherals.svg",
"svg/icons/preferences-desktop-remote-desktop.svg",
"svg/icons/preferences-desktop-screensaver.svg",
"svg/icons/preferences-desktop-sound.svg",
"svg/icons/preferences-desktop-theme.svg",
"svg/icons/preferences-desktop-wallpaper.svg",
"svg/icons/preferences-desktop.svg",
"svg/icons/preferences-system-network-proxy.svg",
"svg/icons/preferences-system-session.svg",
"svg/icons/preferences-system-windows.svg",
"svg/icons/preferences-system.svg",
"svg/icons/printer-error.svg",
"svg/icons/printer.svg",
"svg/icons/process-stop.svg",
"svg/icons/start-here.svg",
"svg/icons/system-file-manager.svg",
"svg/icons/system-installer.svg",
"svg/icons/system-lock-screen.svg",
"svg/icons/system-log-out.svg",
"svg/icons/system-search.svg",
"svg/icons/system-shutdown.svg",
"svg/icons/system-software-update.svg",
"svg/icons/system-users.svg",
"svg/icons/tab-new.svg",
"svg/icons/text-html.svg",
"svg/icons/text-x-generic-template.svg",
"svg/icons/text-x-generic.svg",
"svg/icons/text-x-script.svg",
"svg/icons/user-desktop.svg",
"svg/icons/user-home.svg",
"svg/icons/user-trash-full.svg",
"svg/icons/user-trash.svg",
"svg/icons/utilities-system-monitor.svg",
"svg/icons/utilities-terminal.svg",
"svg/icons/video-display.svg",
"svg/icons/video-x-generic.svg",
"svg/icons/view-fullscreen.svg",
"svg/icons/view-refresh.svg",
"svg/icons/view-restore.svg",
};
static const int num_svg_files = sizeof(svg_files)/sizeof(svg_files[0]);

int lookupSVGPath(const char *filename)
{
    for (int i=0; i<num_svg_files; i++) {
        if (!strcmp(svg_files[i], filename)) {
            return i;
        }
    }
    printf("filename %s not listed as a supported SVG file\n", filename);
    exit(1);
    return 0;
}

const char *getSVGFileName(int ndx)
{
    assert(ndx < num_svg_files);
    if (svg_files[ndx][0] == '!') {
        return svg_files[ndx]+1;
    }
    return svg_files[ndx];
}

int advanceSVGFile(int ndx)
{
    ndx = (ndx+1) % num_svg_files;
    return ndx;
}

int reverseSVGFile(int ndx)
{
    ndx = ndx-1;
    if (ndx < 0) {
        ndx = num_svg_files-1;
    }
    return ndx;
}

// Categories for various SVG files
static struct SVGCategory {
    const char *directory;
    const char *submenu_name;
    int submenu;
} svg_category[] = {
    { "!", "Demos...", 0 },
    { "svg/complex/", "Complex...", 0 },
    { "svg/basic/", "Basic...", 0 },
    { "svg/simple/", "Simple...", 0 },
    { "svg/essentials/", "Essentials...", 0 },
    { "svg/ghostscript/", "Ghostscript...", 0 },
    { "svg/characters/", "Characters...", 0 },
    { "svg/pacman/", "Pac Man...", 0 },
    { "svg/rpg/", "Role Playing Game...", 0 },
    { "svg/ms/", "MS...", 0 },
    { "svg/ravg/", "RAVG...", 0 },
    { "svg/icons/", "Icons...", 0 },
    { "svg/flags/", "Flags...", 0 },
    { "svg/misc/", "Misc...", 0 },
    { "svg/test/", "Test...", 0 },
    { "svg/bezier/", "Bezier...", 0 },
};

int initSVGMenus(GLUTMenuFunc svgMenu)
{
    for (size_t i=0; i<countof(svg_category); i++) {
        svg_category[i].submenu = glutCreateMenu(svgMenu);
    }
    int demo_menu = svg_category[0].submenu;
    int svg_file_menu = glutCreateMenu(svgMenu);
    for (size_t i=0; i<countof(svg_category); i++) {
        glutAddSubMenu(svg_category[i].submenu_name, svg_category[i].submenu);
    }
    for (int j=0; j<num_svg_files; j++) {
        bool found_submenu = false;
        const char *svg_file = svg_files[j];
        if (svg_file[0] == '!') {
            svg_file++;
            glutSetMenu(demo_menu);
            glutAddMenuEntry(svg_file, j);
            continue;
        }
        for (size_t i=0; i<countof(svg_category); i++) {
            size_t len = strlen(svg_category[i].directory);
            if (!strncmp(svg_category[i].directory, svg_file, len)) {
                glutSetMenu(svg_category[i].submenu);
                glutAddMenuEntry(svg_file+len, j);
                found_submenu = true;
                continue;
            }
        }
        if (!found_submenu) {
            glutSetMenu(svg_file_menu);
            glutAddMenuEntry(svg_file, j);
        }
    }
    return svg_file_menu;
}

// SVG files to benchmark
const char* benchmarkFiles[] = 
{
    "svg/complex/tiger.svg",  // complex, 239 paths, WITH stroking
    "svg/complex/Welsh_dragon.svg",  // 150 paths, no stroking
    "svg/complex/Celtic_round_dogs.svg",  // complex, but just 1 path, no sroking
    "svg/basic/butterfly.svg",  // just 3 paths, no stroking
    "svg/simple/spikes.svg",  // all line segments, no stroking
    "svg/complex/Coat_of_Arms_of_American_Samoa.svg", // 953 paths, some stroking
    "svg/complex/cowboy.svg",  // complex, 1366 paths, no stroking

};
const int numBenchmarkFiles = sizeof(benchmarkFiles)/sizeof(benchmarkFiles[0]);
