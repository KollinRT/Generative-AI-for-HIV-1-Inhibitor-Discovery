SELECT * FROM target_dictionary WHERE tid = 191;
-- Human immunodeficiency virus type 1 protease, RVP domain
    -- FROM binding_sites tid=191.


# building queries
SELECT assays.*
FROM assays
INNER JOIN target_dictionary ON assays.tid = target_dictionary.tid
WHERE target_dictionary.tid = 191;

SELECT *
FROM assay_type;

CREATE VIEW named_assays AS
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_id, a.src_assay_id, -- List all columns from "assays" except "assay_type"
       a.chembl_id,
       a.src_id
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
WHERE a.tid = 191;
-- lots of src_id are 1, so from scientific literature.

-- I think I could get the column of it, make a csv, use the csv to query a bunch of pages from chembl?
-- https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services

-- Need to get the CHEMBL ID to scrape the molecules from the CHEMBL website. I need the formulas to feed into my model!


# Other Point
SELECT * FROM target_dictionary WHERE tid = 109928;

SELECT assays.*
FROM assays
INNER JOIN target_dictionary ON assays.tid = target_dictionary.tid
WHERE target_dictionary.tid = 109928;

CREATE VIEW named_assays2 AS
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id, -- List all columns from "assays" except "assay_type"
       a.chembl_id,
       a.src_id
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
WHERE a.tid = 109928;


-- compound_structures molregno (has canonical smiles)-> activities -> idk

-- compound_structures -> (molregno) moleculae_dictionary -> (chembl_id) chembl_id_lookup (chembl_id)?

# write the logical flow of events to get to the molecules that I want to get to canonical smiles to convert to selfies or group selfies...
# REALLY HAVE TO SKETCH IT! WRITE IT UNDER FABRY FLOW OF EVENTS...

# figure out src_id is...

# Got a series of tid by going assays (assay_organism) -> HIV/human immunodef virus lookup.
# going to go from there to looking into the assay results...
# LOOK UP THE OTHER HIV DATABASE AND SPECIFY THE TARGETS FOUND WHEN THEY DID THE TESTS. THEN EMAIL FABRY ASAP!

SELECT DISTINCT TID, assay_organism
FROM assays
WHERE LOWER(assay_organism) LIKE '%hiv%' OR LOWER(assay_organism) LIKE '%human immuno%'; # make it case-insensitive by lowercasing the rows...


SELECT assays.*
FROM assays
INNER JOIN target_dictionary ON assays.tid = target_dictionary.tid
WHERE target_dictionary.tid = 109928;

# combine query with named_assays2
CREATE VIEW named_assays3 AS
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id, -- List all columns from "assays" except "assay_type"
       a.chembl_id,
       a.src_id
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
WHERE a.tid IN (
    SELECT DISTINCT TID
    FROM assays
    WHERE LOWER(assay_organism) LIKE '%hiv%' OR LOWER(assay_organism) LIKE '%human immuno%'
);
# returns 343,851 rows... whoa. That is a bit... Look at domain they target to filter ones of interest!
# This would be the tid (assays) -> binding_sites using tid as an inner join...


SELECT CONCAT(type, ' ', relation, ' ', value, ' ', units) AS concatenated_value
FROM activities
WHERE type LIKE "IC50";

CREATE VIEW named_assays3 AS
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.type, ' ', act.relation, ' ', act.value, ' ', act.units) AS concatenated_value
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id AND act.type LIKE 'IC50' AND act.value IS NOT NULL
WHERE a.tid IN (
    SELECT DISTINCT TID
    FROM assays
    WHERE LOWER(assay_organism) LIKE '%hiv%' OR LOWER(assay_organism) LIKE '%human immuno%'
);

CREATE VIEW named_assays3 AS
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.type, ' ', act.relation, ' ', act.value, ' ', act.units) AS concatenated_value,
       bs.site_name  -- Include the relevant column(s) from the binding_sites table
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id AND act.type LIKE 'IC50' AND act.value IS NOT NULL
RIGHT JOIN binding_sites bs ON a.tid = bs.tid  -- Assuming it's a LEFT JOIN, adjust as needed
WHERE a.tid IN (
    SELECT DISTINCT TID
    FROM assays
    WHERE LOWER(assay_organism) LIKE '%hiv%' OR LOWER(assay_organism) LIKE '%human immuno%'
);

# returns 13,564 rows....
# filtered out cercopithecidae (old world monkeys) and influenza A virus from assay_organism and yielded 13515 records but there are blanks? SO maybe not great?

# capture sources...
# look to see what chembl_id, assay_id, doc_id, # cross reference papers and see what the tests isolate...

# how to deal with < > = in dataset for that? > really isn't appreciable...
    # get counts on frequency of these points that are not =...

# look at how many structures in each category of the assay_organism there are....  and site_name... there are. can do in excel...
# look at integrase enzyme bioassayed for... kinda domain <-> target construed the same...
### majority of what had by them in aspartic protease....
    # kind of decide what to do here... is it for which enzyme is good for???

# look at what to do here....

# # find out smiles for structures here...

SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.type, ' ', act.value, ' ', act.units) AS concatenated_value,
       act.relation as relation,
       bs.site_name  -- Include the relevant column(s) from the binding_sites table
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id AND act.type LIKE 'IC50' AND act.value IS NOT NULL
RIGHT JOIN binding_sites bs ON a.tid = bs.tid  -- Assuming it's a LEFT JOIN, adjust as needed
WHERE a.tid IN (
    SELECT DISTINCT TID
    FROM assays
    WHERE LOWER(assay_organism) LIKE '%hiv%' OR LOWER(assay_organism) LIKE '%human immuno%'
);

# using filtering with MS Excel
# get it just human immunodeficiency virus 1 under assay_organism. we get 9028, filter to "=" only, 7103.
# looking at Human immunodeficiency virus type 1 integrase, rve domain 4766
# looking at Human immunodeficiency virus type 1 protease, RVP domain AND Protease, RVP domain we get 2332.

# filtering for human immunodeficiency virus 1
#chembl_id_lookup...
# compound_properties
#compound_properties cxlogp has it calculated ChemAxon
# STRUCTURES IN compound_structures... they give canonical smiles!

# grading criteria in confidence_score_lookup
/*
0,Default value - Target unknown or has yet to be assigned,Unassigned
1,Target assigned is non-molecular,Non-molecular
2,Target assigned is subcellular fraction,Subcellular fraction
3,Target assigned is molecular non-protein target,Molecular (non-protein)
4,Multiple homologous protein targets may be assigned,Multiple homologous proteins
5,Multiple direct protein targets may be assigned,Multiple proteins
6,Homologous protein complex subunits assigned,Homologous protein complex
7,Direct protein complex subunits assigned,Protein complex
8,Homologous single protein target assigned,Homologous protein
9,Direct single protein target assigned,Protein
*/
# curated_by scores in curation_lookup
/*
Autocuration,Curated against extractor target assignment
Expert,Curated against ChEMBL target assignment from original publication
Intermediate,Curated against ChEMBL target assignment from assay description

 */

SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.type, ' ', act.value, ' ', act.units) AS concatenated_value,
       act.relation AS relation,
       bs.site_name,
       cs.canonical_smiles
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id
LEFT JOIN binding_sites bs ON a.tid = bs.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE a.tid IN (191, 12456) AND
      act.type LIKE 'IC50' AND
      act.value IS NOT NULL AND
      act.relation = '=' AND
      (LOWER(a.assay_organism) LIKE 'human immunodeficiency virus%' OR
       LOWER(a.assay_organism) LIKE 'hiv%')


# Cleaned TODO: Good
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.value) AS IC50,
       act.relation AS relation,
       bs.site_name,
       cs.canonical_smiles
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id
LEFT JOIN binding_sites bs ON a.tid = bs.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE a.tid IN (191, 12456) AND
      act.type LIKE 'IC50' AND
      act.value IS NOT NULL AND
      act.relation = '=' AND
      (LOWER(a.assay_organism) LIKE 'human immunodeficiency virus%' OR
       LOWER(a.assay_organism) LIKE 'hiv%')

# TODO: Try chembl_34;
use chembl_34;
SELECT a.assay_id, a.doc_id, a.description,
       t.assay_desc AS assay_type,
       a.assay_organism, a.tid, a.confidence_score, a.curated_by, a.src_assay_id,
       a.chembl_id,
       a.src_id,
       CONCAT(act.value) AS IC50,
       act.relation AS relation,
       bs.site_name,
       cs.canonical_smiles
FROM assays a
JOIN assay_type t ON a.assay_type = t.assay_type
JOIN activities act ON a.assay_id = act.assay_id
LEFT JOIN binding_sites bs ON a.tid = bs.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
WHERE a.tid IN (191, 12456) AND
      act.type LIKE 'IC50' AND
      act.value IS NOT NULL AND
      act.relation = '=' AND
      (LOWER(a.assay_organism) LIKE 'human immunodeficiency virus%' OR
       LOWER(a.assay_organism) LIKE 'hiv%')


use chembl_33;
SELECT DISTINCT cs.canonical_smiles
FROM compound_structures cs
JOIN activities a ON cs.molregno = a.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
WHERE ass.tid NOT IN (191, 12456)
AND a.standard_type = 'IC50';



# Attempt to do more Druglike molecules
use chembl_34;
SELECT DISTINCT cs.canonical_smiles
FROM compound_structures cs
JOIN activities a ON cs.molregno = a.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
WHERE ass.tid NOT IN (191, 12456) # not HIV-1 inhibs
AND a.standard_type = 'IC50'; # for IC50 values

