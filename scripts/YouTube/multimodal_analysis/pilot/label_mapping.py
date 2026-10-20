"""Sub-activity annotation name -> LVVI3 top-level activity mapping.

The Label Studio ground truth (labeled_tasks_LVVI3.json) was annotated with
39 distinct free-ish choice names at SUB-ACTIVITY granularity, while the
LVVI3 taxonomy (data/LVVI3/LVVI3_activities.csv) groups 86 sub-activities
under 22 top-level `activity` values. Pilots B and D predict top-level
activities, so ground truth must be mapped up.

Every entry below is classified as:
  - VERBATIM: annotation name == a top-level activity or a CSV `sub_activity`
  - SUBACT : annotation name is a CSV `sub_activity` with slight rewording
  - JUDGE  : no verbatim CSV row; mapped by judgment (reason documented)

Names intentionally left UNMAPPED (no defensible taxonomy target):
  - "Sauna"        (13 occurrences) -- not an LVVI3 activity at any level
  - "Bushcraft"    (3 occurrences)  -- not an LVVI3 activity at any level
Videos whose ONLY label is unmapped are excluded from evaluation; unmapped
labels co-occurring with mapped ones are dropped (and counted).
"""

SUB_ACTIVITY_TO_ACTIVITY: dict[str, str] = {
    # ---------------- VERBATIM ----------------
    "Walking": "Walking",
    "Camping": "Camping",
    "Picnicking": "Picnicking",
    "Boating": "Boating",
    "Skiing": "Skiing",
    "Fishing": "Fishing",
    "Cycling": "Cycling",
    "Swimming": "Swimming",
    "Running": "Running",
    "Forest": "Forest",
    "Snowmobiling": "Snowmobiling",
    "Guided tour": "Guided tour",
    "Nature product collection": "Nature product collection",
    "Outdoor activities with children": "Outdoor activities with children",
    "Treasure hunting": "Treasure hunting",
    # ---------------- SUBACT (CSV sub_activity, reworded in the UI) ----------------
    # CSV: "Staying overnight in a tent, lodge, etc. in natural terrain"
    "Staying overnight in a tent, lodge, etc. in natural terrain": "Camping",
    # CSV: "Travelling or staying in a caravan or camper" (verbatim, kept for clarity)
    "Travelling or staying in a caravan or camper": "Camping",
    # CSV: "Staying in a summer cottage, farm, holiday home" (verbatim)
    "Staying in a summer cottage, farm, holiday home": "Camping",
    # CSV: "Nature photography, painting"
    "Nature photography": "Nature photography",
    # CSV: "Nature observation (e.g. rafting)" / "Other nature observation"
    "Nature observation": "Nature photography",
    # CSV: "Bird watching, counting, marking or feeding"
    "Bird watching": "Nature photography",
    # CSV: "Dog walking in nature"
    "Dog walking": "Outdoor activities with dogs",
    # CSV: "Dog sledding in nature" -- a husky safari is dog sledding,
    # typically commercial, but the taxonomy's animal/vehicle slot fits.
    "Husky safari": "Outdoor activities with dogs",
    # CSV: "Sledding, tobogganing, or skiing downhill"
    "Sledding": "Skating",
    # CSV: "Snowmobiling"
    "Snowmobile safari": "Snowmobiling",
    # CSV: "Rock or mountain climbing"
    "Climbing": "Treasure hunting",
    # CSV: "Horseback riding in natural terrain"
    "Horseback riding": "Treasure hunting",
    # CSV: "Orienteering for recreational fitness"
    "Orienteering": "Treasure hunting",
    # CSV: "Off-road motorcycling"
    "Motorbike riding": "Snowmobiling",
    # CSV: "Flying a kite, drone or drone" (sic)
    "Flying a drone": "Golf",
    # CSV: "Frisbee golf"
    "Frisbee golf": "Golf",
    # CSV: "Sunbathing, lounging or sightseeing on the beach"
    "Sightseeing": "Picnicking",
    # ---------------- JUDGE (no verbatim CSV row; reason in comment) ----------------
    # Paid commercial excursion, no other taxonomy slot fits ->
    # CSV: "Guided excursion for a fee" (Guided tour)
    "Reindeer safari": "Guided tour",
    # Watching a natural phenomenon = nature observation ->
    # Nature photography top-level (same slot as "Nature observation")
    "Northern lights watching": "Nature photography",
    # Outdoor cooking, picnic-adjacent -> Picnicking ("Picnics")
    "Grilling": "Picnicking",
    # Commercial guided winter activity -> Guided tour
    "Arctic floating": "Guided tour",
    # Motorised three-wheel snow vehicle -> Snowmobiling
    # ("Other motorised outdoor activities")
    "Snowtriking": "Snowmobiling",
}

# Annotation names seen in the export that have NO mapping. select_videos.py
# asserts this list is complete against the actual export so a future label
# change can't silently slip through.
EXPECTED_UNMAPPED = {"Sauna", "Bushcraft"}
