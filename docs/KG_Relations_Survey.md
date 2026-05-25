# Survey: Inter-Entity Relations in Knowledge Graphs
## For KG-Grounded Hallucination Mitigation in LLMs (Environmental Sustainability / Statistical Domain)

*Compiled: April 2026*

---

## Table of Contents
1. [Per-Source Relation Inventories](#1-per-source-relation-inventories)
2. [Consolidated Universal Relation Taxonomy](#2-consolidated-universal-relation-taxonomy)
3. [Cross-KG Frequency Matrix](#3-cross-kg-frequency-matrix)
4. [Storage Tools & Formats Comparison](#4-storage-tools--formats-comparison)
5. [Notable Findings & Recommendations](#5-notable-findings--recommendations)

---

## 1. Per-Source Relation Inventories

### 1.1 ConceptNet (34 relations)

**Source:** [ConceptNet 5.5 Relations Wiki](https://github.com/commonsense/conceptnet5/wiki/Relations)  
**Scale:** ~21M edges, 8M nodes, 304 languages  
**Paper:** Speer, Chin & Havasi. "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." AAAI 2017.

ConceptNet defines **34 canonical relations** (camel-cased, language-neutral):

| # | Relation | Type | Symmetric? | Example |
|---|----------|------|------------|---------|
| 1 | `/r/RelatedTo` | Associative | ✓ | learn ↔ erudition |
| 2 | `/r/FormOf` | Linguistic | | slept → sleep |
| 3 | `/r/IsA` | Taxonomic | | car → vehicle |
| 4 | `/r/PartOf` | Mereological | | gearshift → car |
| 5 | `/r/HasA` | Possessive | | bird → wing |
| 6 | `/r/UsedFor` | Functional | | bridge → cross water |
| 7 | `/r/CapableOf` | Functional | | knife → cut |
| 8 | `/r/AtLocation` | Spatial | | butter → refrigerator |
| 9 | `/r/Causes` | Causal | | exercise → sweat |
| 10 | `/r/HasSubevent` | Temporal | | eating → chewing |
| 11 | `/r/HasFirstSubevent` | Temporal | | sleep → close eyes |
| 12 | `/r/HasLastSubevent` | Temporal | | cook → clean up kitchen |
| 13 | `/r/HasPrerequisite` | Causal/Temporal | | dream → sleep |
| 14 | `/r/HasProperty` | Descriptive | | ice → cold |
| 15 | `/r/MotivatedByGoal` | Purposive | | compete → win |
| 16 | `/r/ObstructedBy` | Causal | | sleep → noise |
| 17 | `/r/Desires` | Purposive | | person → love |
| 18 | `/r/CreatedBy` | Provenance | | cake → bake |
| 19 | `/r/Synonym` | Equivalence | ✓ | sunlight ↔ sunshine |
| 20 | `/r/Antonym` | Comparative | ✓ | black ↔ white |
| 21 | `/r/DistinctFrom` | Comparative | ✓ | red ↔ blue |
| 22 | `/r/DerivedFrom` | Etymological | | pocketbook → book |
| 23 | `/r/SymbolOf` | Associative | | red → fervor |
| 24 | `/r/DefinedAs` | Equivalence | | peace → absence of war |
| 25 | `/r/MannerOf` | Taxonomic | | auction → sale |
| 26 | `/r/LocatedNear` | Spatial | ✓ | chair ↔ table |
| 27 | `/r/HasContext` | Contextual | | astern → ship |
| 28 | `/r/SimilarTo` | Associative | ✓ | mixer ↔ food processor |
| 29 | `/r/EtymologicallyRelatedTo` | Etymological | ✓ | folkmusiikki ↔ folk music |
| 30 | `/r/EtymologicallyDerivedFrom` | Etymological | | dejta → date |
| 31 | `/r/CausesDesire` | Causal/Purposive | | having no food → go to store |
| 32 | `/r/MadeOf` | Compositional | | bottle → plastic |
| 33 | `/r/ReceivesAction` | Functional | | button → push |
| 34 | `/r/ExternalURL` | Meta/Linking | | (links to DBpedia, etc.) |

**Deprecated but notable:** `/r/InstanceOf` (merged into IsA), `/r/Entails`, `/r/NotIsA`, `/r/NotDesires`, `/r/NotUsedFor`, `/r/NotCapableOf`, `/r/NotHasProperty`.

---

### 1.2 Wikidata (Top inter-entity properties)

**Source:** [Wikidata Top 100 Properties](https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/Top100) (data as of 2026-04-25)  
**Scale:** 1.65 billion statements (semantic triples), 100M+ items, 12,000+ properties  
**Format:** Item (Q-ID) → Property (P-ID) → Value (Q-ID or literal)

Top **inter-entity** (entity→entity) properties by usage count:

| Rank | Property (P-code) | Usage Count | Category |
|------|-------------------|-------------|----------|
| 1 | `P31` instance of | 126,913,721 | Taxonomic |
| 2 | `P17` country | 20,669,062 | Spatial |
| 3 | `P131` located in admin. territorial entity | 15,303,002 | Spatial |
| 4 | `P106` occupation | 13,745,739 | Classificatory |
| 5 | `P21` sex or gender | 11,232,557 | Classificatory |
| 6 | `P735` given name | 8,804,562 | Associative |
| 7 | `P703` found in taxon | 7,267,117 | Taxonomic |
| 8 | `P59` constellation | 7,371,494 | Spatial |
| 9 | `P27` country of citizenship | 6,283,860 | Spatial |
| 10 | `P734` family name | 6,195,707 | Associative |
| 11 | `P195` collection | 5,829,189 | Mereological |
| 12 | `P361` **part of** | 5,497,760 | Mereological |
| 13 | `P279` **subclass of** | 5,179,670 | Taxonomic |
| 14 | `P276` location | 5,165,269 | Spatial |
| 15 | `P171` parent taxon | 3,960,201 | Taxonomic |
| 16 | `P155` follows | 3,131,787 | Temporal/Sequential |
| 17 | `P156` followed by | 3,094,794 | Temporal/Sequential |
| 18 | `P495` country of origin | 2,956,111 | Spatial |
| 19 | `P527` **has part(s)** | 2,798,187 | Mereological |
| 20 | `P50` author | 35,896,956 | Provenance |
| 21 | `P1433` published in | 46,632,864 | Provenance |
| 22 | `P921` main subject | 34,738,986 | Associative |
| 23 | `P69` educated at | 3,640,042 | Associative |
| 24 | `P54` member of sports team | 3,736,647 | Associative |
| 25 | `P19` place of birth | 4,238,312 | Spatial |

**Additional important inter-entity properties (beyond top 100):**

| Property | P-code | Category |
|----------|--------|----------|
| `has cause` | P828 | Causal |
| `has effect` | P1542 | Causal |
| `influenced by` | P737 | Causal |
| `opposite of` | P461 | Comparative |
| `different from` | P1889 | Comparative |
| `said to be the same as` | P460 | Equivalence |
| `has use` | P366 | Functional |
| `contains admin. territorial entity` | P150 | Spatial (inverse of P131) |
| `shares border with` | P47 | Spatial |
| `significant event` | P793 | Temporal |
| `facet of` | P1269 | Taxonomic |
| `replaced by` | P1366 | Temporal |
| `replaces` | P1365 | Temporal |

---

### 1.3 DBpedia (~760 ontology classes, ~2,800 properties)

**Source:** [DBpedia Ontology](https://dbpedia.org/ontology/)  
**Scale:** 9.5 billion RDF triples (2016 release), 850M+ triples (2021), 6M entities  
**Ontology:** 760 classes, ~2,800 properties derived from Wikipedia infoboxes

Key **inter-entity (ObjectProperty)** relations in the DBpedia ontology:

| Relation | DBpedia URI | Category |
|----------|-------------|----------|
| `rdf:type` / `dbo:type` | Instance typing | Taxonomic |
| `rdfs:subClassOf` | Class hierarchy | Taxonomic |
| `dbo:location` / `dbo:city` / `dbo:country` | Location | Spatial |
| `dbo:isPartOf` | Part-whole | Mereological |
| `dbo:part` | Has-part | Mereological |
| `dbo:genre` | Classification | Associative |
| `dbo:author` | Authorship | Provenance |
| `dbo:creator` | Creation | Provenance |
| `dbo:publisher` | Publishing | Provenance |
| `dbo:birthPlace` / `dbo:deathPlace` | Place of event | Spatial |
| `dbo:nationality` | Citizenship | Spatial |
| `dbo:spouse` / `dbo:parent` / `dbo:child` | Kinship | Associative |
| `dbo:predecessor` / `dbo:successor` | Sequence | Temporal |
| `dbo:influencedBy` / `dbo:influenced` | Influence | Causal |
| `dbo:knownFor` | Achievement | Associative |
| `dbo:league` / `dbo:team` | Membership | Associative |
| `dbo:associatedBand` / `dbo:associatedMusicalArtist` | Association | Associative |
| `dbo:product` / `dbo:industry` | Domain | Functional |
| `dbo:material` | Composition | Compositional |
| `owl:sameAs` | Identity | Equivalence |
| `dbo:wikiPageWikiLink` | General link | Associative |

**Note:** DBpedia uses `owl:sameAs` extensively to link to Wikidata, Freebase, YAGO, and other LOD datasets (45M+ interlinks).

---

### 1.4 YAGO (Schema.org-based taxonomy)

**Source:** [YAGO Knowledge Base](https://yago-knowledge.org/)  
**Scale:** YAGO3: 120M+ facts, 10M+ entities; YAGO4: based on Wikidata + Schema.org  
**Accuracy:** >95% (manually evaluated)

YAGO4/4.5 uses **Schema.org relationship designators** combined with Wikidata data:

| Relation | Source Vocabulary | Category |
|----------|-------------------|----------|
| `rdf:type` | RDF | Taxonomic |
| `rdfs:subClassOf` | RDFS | Taxonomic |
| `schema:isPartOf` | Schema.org | Mereological |
| `schema:hasPart` | Schema.org | Mereological |
| `schema:location` / `schema:containedInPlace` | Schema.org | Spatial |
| `schema:containsPlace` | Schema.org | Spatial |
| `schema:memberOf` | Schema.org | Associative |
| `schema:knows` | Schema.org | Associative |
| `schema:author` | Schema.org | Provenance |
| `schema:creator` | Schema.org | Provenance |
| `schema:birthPlace` / `schema:deathPlace` | Schema.org | Spatial |
| `schema:nationality` | Schema.org | Spatial |
| `schema:spouse` / `schema:parent` / `schema:children` | Schema.org | Kinship |
| `schema:sameAs` | Schema.org | Equivalence |
| `schema:about` | Schema.org | Associative |
| `schema:subjectOf` | Schema.org | Associative |
| `schema:material` | Schema.org | Compositional |
| `schema:category` | Schema.org | Classificatory |

YAGO3 originally derived its taxonomy from WordNet synsets (hyponymy/hypernymy) and linked to SUMO ontology.

---

### 1.5 Freebase (Type/Property system)

**Source:** [Freebase](https://en.wikipedia.org/wiki/Freebase_(database)) (discontinued 2016, migrated to Wikidata)  
**Scale:** 44M topics, 2.4B facts (at peak, Jan 2014)  
**Backend:** proprietary graph DB called `graphd`, queried via MQL (both open-sourced)

Freebase used a **domain/type/property** system rather than a fixed relation ontology:

| Domain | Example Type | Example Properties (Entity→Entity) |
|--------|-------------|-------------------------------------|
| People | `/people/person` | `/people/person/place_of_birth`, `/people/person/nationality`, `/people/person/spouse_s` |
| Location | `/location/location` | `/location/location/containedby`, `/location/location/contains`, `/location/location/adjoin_s` |
| Organization | `/organization/organization` | `/organization/organization/headquarters`, `/organization/organization/parent`, `/organization/organization/child` |
| Biology | `/biology/organism` | `/biology/organism/organism_type`, `/biology/organism/higher_classification` |
| Film/Music | `/film/film` | `/film/film/directed_by`, `/film/film/produced_by`, `/film/film/genre` |
| Common | `/common/topic` | `/common/topic/alias`, `/type/object/type` (≈ `instanceOf`) |

**Structural inter-entity patterns from Freebase:**
- **instanceOf** (type assignment)
- **containedBy / contains** (spatial/mereological)
- **partOf / hasPart**
- **subclassOf** (type hierarchy)
- **influences / influencedBy**
- **relatedTo** (cross-domain links)

---

### 1.6 Microsoft GraphRAG (LLM-extracted relations)

**Source:** [GraphRAG Documentation](https://microsoft.github.io/graphrag/), [Arxiv paper 2404.16130](https://arxiv.org/pdf/2404.16130)  
**Approach:** Uses GPT-4/LLM prompts to extract entities & relationships from raw text

GraphRAG does **not** use a fixed relation ontology. Instead, it:

1. **Extracts named entities** (persons, organizations, places, concepts, events)
2. **Extracts relationships** as free-text descriptions between entity pairs, with a **strength score** (1-10)
3. **Builds a knowledge graph** where edges have natural-language descriptions
4. **Applies Leiden community detection** for hierarchical clustering
5. **Generates community summaries** used for RAG at query time

**Typical extracted relation types** (LLM-generated, not from a fixed schema):
- LOCATED_IN, HEADQUARTERED_IN
- PART_OF, MEMBER_OF, BELONGS_TO
- WORKS_FOR, EMPLOYED_BY
- AUTHORED_BY, CREATED_BY
- CAUSES, LEADS_TO, RESULTS_IN
- RELATED_TO, ASSOCIATED_WITH
- OPPOSES, SUPPORTS
- PRECEDED_BY, FOLLOWED_BY, SUCCEEDED_BY

**Key insight for our work:** GraphRAG demonstrates that LLM-based extraction naturally produces relations that mirror the universal categories found across fixed-schema KGs. This validates designing a fixed schema for our domain.

---

### 1.7 Schema.org / Google Knowledge Graph (~800 types, ~1,400+ properties)

**Source:** [Schema.org Full Hierarchy](https://schema.org/docs/full.html) (V30.0, 2026-03-19)  
**Maintained by:** Google, Microsoft, Yahoo, Yandex (W3C Community Group)  
**Google KG:** Powered by Freebase data + Schema.org vocabulary

**Top-level type hierarchy (inter-entity "Thing" types):**
- Thing → Action, BioChemEntity, CreativeWork, Event, Intangible, MedicalEntity, Organization, Person, Place, Product, Taxon

**Key inter-entity (Thing→Thing) properties in Schema.org:**

| Property | Domain → Range | Category |
|----------|---------------|----------|
| `isPartOf` | CreativeWork/Place → CreativeWork/Place | Mereological |
| `hasPart` | CreativeWork/Place → CreativeWork/Place | Mereological |
| `containedInPlace` | Place → Place | Spatial |
| `containsPlace` | Place → Place | Spatial |
| `location` | Event/Org → Place | Spatial |
| `memberOf` | Person/Org → Organization | Associative |
| `subOrganization` | Organization → Organization | Taxonomic |
| `parentOrganization` | Organization → Organization | Taxonomic |
| `author` / `creator` | CreativeWork → Person/Org | Provenance |
| `about` | CreativeWork → Thing | Associative |
| `subjectOf` | Thing → CreativeWork | Associative |
| `sameAs` | Thing → Thing | Equivalence |
| `knows` | Person → Person | Associative |
| `spouse` / `parent` / `children` | Person → Person | Kinship |
| `birthPlace` / `deathPlace` | Person → Place | Spatial |
| `material` | Product → Text/Thing | Compositional |
| `category` | Thing → Thing/Text | Classificatory |
| `isRelatedTo` | Product → Product/Service | Associative |
| `isSimilarTo` | Product → Product/Service | Comparative |
| `supersededBy` | Thing → Thing | Temporal |

---

### 1.8 SKOS (W3C Vocabulary for Knowledge Organization)

**Source:** [SKOS Reference W3C Recommendation](https://www.w3.org/TR/skos-reference/) (2009-08-18)  
**Purpose:** Bridging informal KOS (thesauri, taxonomies, classification schemes) with the Semantic Web

**Core inter-concept semantic relations:**

| Relation | URI | Category | Properties |
|----------|-----|----------|------------|
| `skos:broader` | Concept → Concept | Taxonomic/Hierarchical | Not transitive |
| `skos:narrower` | Concept → Concept | Taxonomic/Hierarchical | Inverse of broader |
| `skos:broaderTransitive` | Concept → Concept | Taxonomic/Hierarchical | Transitive |
| `skos:narrowerTransitive` | Concept → Concept | Taxonomic/Hierarchical | Transitive |
| `skos:related` | Concept → Concept | Associative | Symmetric, not transitive |
| `skos:broadMatch` | Concept → Concept | Cross-scheme hierarchical | |
| `skos:narrowMatch` | Concept → Concept | Cross-scheme hierarchical | |
| `skos:relatedMatch` | Concept → Concept | Cross-scheme associative | Symmetric |
| `skos:exactMatch` | Concept → Concept | Equivalence | Transitive, symmetric |
| `skos:closeMatch` | Concept → Concept | Equivalence | Symmetric, not transitive |

**Key design principle:** SKOS enforces disjointness between hierarchical (`broaderTransitive`) and associative (`related`) links — a concept pair cannot be both.

---

### 1.9 PharmKG (Biomedical Reference)

**Source:** Zheng et al., "PharmKG: a dedicated knowledge graph benchmark for bomedical data mining." Briefings in Bioinformatics, 2021.  
**Scale:** 500K+ nodes, 29 relation types, 3 entity categories (Gene, Disease, Drug)

**29 relations across 3 entity categories:**

| Relation Category | Examples |
|-------------------|----------|
| Gene-Disease | `associated_with`, `biomarker_of`, `causal_mutation_of`, `therapeutic_target_for` |
| Drug-Disease | `indicated_for`, `contraindicated_for`, `side_effect_of`, `off_label_use_for` |
| Drug-Gene | `target_of`, `transporter_of`, `enzyme_of`, `carrier_of` |
| Gene-Gene | `interacts_with`, `regulates`, `co-expressed_with` |

**Key insight:** Domain-specific KGs like PharmKG use highly specialized entity-to-entity relations, but they still fall into universal categories (Causal, Associative, Functional).

---

### 1.10 Environmental / Sustainability KGs

**Notable domain-specific KGs:**

| KG | Domain | Key Relations | Source |
|----|--------|---------------|--------|
| **ENVO** (Environment Ontology) | Environment/Ecology | `part_of`, `located_in`, `has_habitat`, `derives_from`, `adjacent_to`, `causally_upstream_of` | OBO Foundry |
| **SWEET** (Semantic Web for Earth & Environmental Terminology) | Earth Science | `hasPart`, `partOf`, `contains`, `isContainedIn`, `hasRole`, `hasImpact`, `causes`, `hasSource` | NASA JPL |
| **SDG Ontology** | Sustainable Development Goals | `contributes_to`, `impedes`, `measured_by`, `targets`, `related_to` | UN/Academic |
| **GeoNames** | Geographical entities | `parentFeature`, `locatedIn`, `neighbourOf`, `nearBy` | Creative Commons |
| **EcoLexicon** | Environmental science | `causes`, `affects`, `is_a`, `part_of`, `takes_place_in`, `result_of`, `produced_by` | University of Granada |

---

## 2. Consolidated Universal Relation Taxonomy

Based on cross-referencing all 9 KG sources, the following **28 universal inter-entity relation types** emerge, grouped into 7 categories:

### 2.1 Taxonomic / Hierarchical Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 1 | **IsA / InstanceOf** | ConceptNet, Wikidata (P31), DBpedia (rdf:type), YAGO, Freebase, Schema.org, SKOS (broader) | **7/9 = UNIVERSAL** |
| 2 | **SubclassOf / BroaderThan** | Wikidata (P279), DBpedia (rdfs:subClassOf), YAGO, SKOS (skos:broader), OWL, Schema.org | **6/9 = UNIVERSAL** |
| 3 | **NarrowerThan / HasSubtype** | SKOS (skos:narrower), Wikidata (inverse P279), DBpedia, YAGO | 4/9 = UNIVERSAL |
| 4 | **MannerOf** | ConceptNet | 1/9 |

### 2.2 Mereological / Part-Whole Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 5 | **PartOf** | ConceptNet, Wikidata (P361), DBpedia, YAGO, Schema.org, Freebase, ENVO, SWEET | **8/9 = UNIVERSAL** |
| 6 | **HasPart / Contains** | Wikidata (P527), DBpedia, YAGO, Schema.org, Freebase, SWEET | **6/9 = UNIVERSAL** |
| 7 | **MadeOf / HasMaterial** | ConceptNet, Schema.org, DBpedia | 3/9 = UNIVERSAL |
| 8 | **MemberOf** | Wikidata, Schema.org, DBpedia, Freebase | 4/9 = UNIVERSAL |

### 2.3 Causal / Temporal Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 9 | **Causes / HasEffect** | ConceptNet, Wikidata (P828/P1542), GraphRAG, ENVO, SWEET, PharmKG | **6/9 = UNIVERSAL** |
| 10 | **HasPrerequisite / DependsOn** | ConceptNet, GraphRAG | 2/9 |
| 11 | **Precedes / Follows** | Wikidata (P155/P156), DBpedia (predecessor/successor), Schema.org, GraphRAG | **4/9 = UNIVERSAL** |
| 12 | **HasSubevent** | ConceptNet | 1/9 |
| 13 | **InfluencedBy / Influences** | Wikidata (P737), DBpedia, GraphRAG | 3/9 = UNIVERSAL |
| 14 | **ReplacedBy / Replaces** | Wikidata (P1365/P1366), Schema.org (supersededBy) | 2/9 |

### 2.4 Spatial / Locational Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 15 | **LocatedIn / ContainedInPlace** | ConceptNet (AtLocation), Wikidata (P131/P276), DBpedia, YAGO, Schema.org, Freebase, GeoNames | **7/9 = UNIVERSAL** |
| 16 | **ContainsPlace** | Wikidata (P150), Schema.org, Freebase, GeoNames | 4/9 = UNIVERSAL |
| 17 | **AdjacentTo / SharesBorderWith** | Wikidata (P47), GeoNames, ENVO | 3/9 = UNIVERSAL |
| 18 | **LocatedNear** | ConceptNet, GeoNames | 2/9 |

### 2.5 Associative / Semantic Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 19 | **RelatedTo / AssociatedWith** | ConceptNet, SKOS (skos:related), GraphRAG, DBpedia (wikiPageWikiLink) | **4/9 = UNIVERSAL** |
| 20 | **SimilarTo** | ConceptNet, Schema.org (isSimilarTo) | 2/9 |
| 21 | **HasContext / MainSubject** | ConceptNet, Wikidata (P921) | 2/9 |
| 22 | **CreatedBy / Author** | ConceptNet, Wikidata (P50), DBpedia, YAGO, Schema.org, Freebase | **6/9 = UNIVERSAL** |

### 2.6 Functional / Purposive Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 23 | **UsedFor / HasFunction** | ConceptNet, Wikidata (P366), PharmKG | 3/9 = UNIVERSAL |
| 24 | **CapableOf** | ConceptNet | 1/9 |
| 25 | **Desires / MotivatedByGoal** | ConceptNet | 1/9 |

### 2.7 Comparative / Equivalence Relations

| # | Relation | KG Sources (count) | Status |
|---|----------|-------------------|--------|
| 26 | **SameAs / ExactMatch** | DBpedia (owl:sameAs), SKOS (skos:exactMatch), Schema.org (sameAs), Wikidata (P460) | **4/9 = UNIVERSAL** |
| 27 | **OppositeOf / Antonym** | ConceptNet, Wikidata (P461) | 2/9 |
| 28 | **DistinctFrom / DifferentFrom** | ConceptNet, Wikidata (P1889) | 2/9 |

---

## 3. Cross-KG Frequency Matrix

Relations appearing in **3+ major KGs** are marked as **UNIVERSAL** (★):

| Relation | CN | WD | DB | YA | FB | SO | SK | GR | PhKG | ENVO | Count | Status |
|----------|----|----|----|----|----|----|----|----|------|------|-------|--------|
| **IsA/InstanceOf** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | **8** | ★★★ |
| **SubclassOf** | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | | **6** | ★★★ |
| **PartOf** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | | ✓ | **7** | ★★★ |
| **HasPart** | | ✓ | ✓ | ✓ | ✓ | ✓ | | | | ✓ | **6** | ★★★ |
| **LocatedIn** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | | ✓ | **8** | ★★★ |
| **Causes** | ✓ | ✓ | | | | | | ✓ | ✓ | ✓ | **5** | ★★ |
| **CreatedBy/Author** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | **6** | ★★★ |
| **Follows/Precedes** | | ✓ | ✓ | | | ✓ | | ✓ | | | **4** | ★ |
| **RelatedTo** | ✓ | | ✓ | | | | ✓ | ✓ | | | **4** | ★ |
| **SameAs/ExactMatch** | | ✓ | ✓ | ✓ | | ✓ | ✓ | | | | **5** | ★★ |
| **MemberOf** | | ✓ | ✓ | | ✓ | ✓ | | ✓ | | | **5** | ★★ |
| **MadeOf/Material** | ✓ | | ✓ | | | ✓ | | | | | **3** | ★ |
| **UsedFor** | ✓ | ✓ | | | | | | | ✓ | | **3** | ★ |
| **Influences** | | ✓ | ✓ | | | | | ✓ | | | **3** | ★ |
| **AdjacentTo** | | ✓ | | | | | | | | ✓ | **2** | |
| **SimilarTo** | ✓ | | | | | ✓ | | | | | **2** | |
| **OppositeOf** | ✓ | ✓ | | | | | | | | | **2** | |
| **DistinctFrom** | ✓ | ✓ | | | | | | | | | **2** | |

**Legend:** CN=ConceptNet, WD=Wikidata, DB=DBpedia, YA=YAGO, FB=Freebase, SO=Schema.org, SK=SKOS, GR=GraphRAG, PhKG=PharmKG, ENVO=Environment Ontology

---

## 4. Storage Tools & Formats Comparison

### 4.1 Graph Databases & Triple Stores

| Tool | Type | Query Language | Scalability | Python API | Best For |
|------|------|---------------|-------------|------------|----------|
| **Neo4j** | Property Graph DB | Cypher | Billions of nodes | `neo4j` driver, `py2neo` | Production KGs, complex traversals |
| **Apache Jena / Fuseki** | RDF Triple Store | SPARQL | Large-scale RDF | `SPARQLWrapper` | Standards-compliant Linked Data |
| **Blazegraph** | RDF Triple Store | SPARQL | Used by Wikidata | `SPARQLWrapper` | SPARQL endpoints, Wikidata queries |
| **Virtuoso** | RDF + Relational | SPARQL + SQL | Very large scale | ODBC/SPARQL | DBpedia's backend, enterprise |
| **Apache TinkerPop / Gremlin** | Graph Computing Framework | Gremlin | Distributed | `gremlinpython` | Vendor-agnostic graph traversals |
| **Amazon Neptune** | Managed Graph DB | SPARQL + Gremlin | Cloud-scale | Boto3 | AWS-native, managed service |
| **Stardog** | Knowledge Graph Platform | SPARQL + GraphQL | Enterprise | Python client | Enterprise KG with reasoning |

### 4.2 Python Libraries (In-Memory / Lightweight)

| Library | Type | Format Support | Max Scale | Best For |
|---------|------|---------------|-----------|----------|
| **NetworkX** | General graph library | Custom | ~100K nodes | Prototyping, analysis, small KGs |
| **RDFLib** | RDF library | N-Triples, Turtle, JSON-LD, RDF/XML | ~1M triples | RDF/Linked Data, SPARQL queries |
| **igraph** | High-perf graph lib | Custom | ~10M edges | Fast graph algorithms |
| **DGL** (Deep Graph Library) | GNN framework | CSR, heterogeneous | GPU-scale | GNN-based link prediction, node classification |
| **PyG** (PyTorch Geometric) | GNN framework | HeteroData | GPU-scale | GNN research, heterogeneous KGs |
| **ChromaDB** | Vector store | Embeddings | ~1M vectors | Embedding-based KG retrieval |

### 4.3 Serialization Formats

| Format | Extension | Standard | Human-Readable | Size Efficiency | Tool Support |
|--------|-----------|----------|----------------|-----------------|--------------|
| **N-Triples** | `.nt` | W3C | Medium | Low (verbose) | Universal RDF support |
| **Turtle** | `.ttl` | W3C | **High** | **Good** (compact) | RDFLib, Jena, all SPARQL stores |
| **JSON-LD** | `.jsonld` | W3C | High (JSON) | Medium | Web APIs, Schema.org, ConceptNet |
| **RDF/XML** | `.rdf` | W3C | Low | Medium | Legacy systems |
| **N-Quads** | `.nq` | W3C | Medium | Low | Named graphs, provenance |
| **JSONL (custom)** | `.jsonl` | De facto | High | **Good** | LLM pipelines, streaming |
| **Parquet** | `.parquet` | Apache | No (binary) | **Excellent** | Analytics, columnar queries |
| **TSV/CSV** | `.tsv` | De facto | High | Good | Simple, YAGO3 uses TSV |

### 4.4 Recommended Stack for This Project

Given our use case (KG-grounded hallucination detection for environmental statistics):

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Triple Storage** | JSONL (for pipeline) + ChromaDB (for retrieval) | Already in use; streaming-friendly |
| **Embedding Index** | ChromaDB | Already integrated in codebase |
| **Graph Analysis** | NetworkX (small) or Neo4j (production) | NetworkX for paper experiments |
| **GNN-ready** | PyG HeteroData export | If GNN-based verification is added |
| **Serialization** | JSONL primary, Turtle for interop | JSONL matches our pipeline |
| **SPARQL endpoint** | Apache Jena Fuseki (if needed) | Standards compliance for reproducibility |

---

## 5. Notable Findings & Recommendations

### 5.1 Key Findings

1. **Only ~10 structural relation types are truly universal** across all major KGs:
   - `IsA/InstanceOf`, `SubclassOf`, `PartOf`, `HasPart`, `LocatedIn`, `Causes`, `CreatedBy/Author`, `RelatedTo`, `SameAs`, `MemberOf`
   - These appear in 4-8 out of 9 surveyed KGs

2. **ConceptNet has the richest commonsense relation vocabulary** (34 types), including unique relations like `CausesDesire`, `ObstructedBy`, `MotivatedByGoal` — valuable for reasoning about LLM hallucinations involving causal/purposive claims

3. **Wikidata is the largest open KG** (1.65B triples) and uses `P31` (instance of) + `P279` (subclass of) as the taxonomic backbone — these two properties alone account for 132M usages

4. **GraphRAG confirms LLMs naturally extract the universal relation categories**, validating a fixed-schema approach for our pipeline

5. **SKOS provides the gold-standard formal framework** for hierarchical vs. associative relation disjointness — relevant for preventing relation confusion in our KG

6. **Environmental KGs** (ENVO, SWEET, EcoLexicon) add domain relations:
   - `hasImpact`, `affects`, `measured_by`, `contributes_to`, `impedes`
   - These extend the Causal category specifically for sustainability

7. **Our existing 46-relation schema** focuses correctly on **entity-to-literal** statistical relations (HasStatistic, HasNumericValue, etc.) — now needs complementary **entity-to-entity** structural relations

### 5.2 Recommendations for the Paper

**Proposed inter-entity relations to add to our schema** (complement the existing 46 statistical relations):

| Priority | Relation | Justification | Universal Count |
|----------|----------|--------------|-----------------|
| **HIGH** | `IsA` | Foundation of all KG typing | 8/9 |
| **HIGH** | `PartOf` / `HasPart` | Environmental topics are hierarchical (ecosystem → biome → habitat) | 7/9 |
| **HIGH** | `LocatedIn` | Geospatial grounding critical for environmental claims | 8/9 |
| **HIGH** | `Causes` / `HasEffect` | Causal claims are #1 hallucination target in env. stats | 6/9 |
| **HIGH** | `RelatedTo` | Catch-all for associative links; fallback relation | 4/9 |
| **MEDIUM** | `SubclassOf` | Taxonomic hierarchy (e.g., Renewable Energy → Solar Energy) | 6/9 |
| **MEDIUM** | `Precedes` / `Follows` | Temporal ordering of events, policies | 4/9 |
| **MEDIUM** | `MeasuredBy` | Links metric → indicator (domain-specific, from SWEET/SDG) | 2/9 |
| **MEDIUM** | `InfluencedBy` | Policy/factor influence chains | 3/9 |
| **LOW** | `SameAs` | Entity alignment across sources | 5/9 |
| **LOW** | `SimilarTo` | Near-equivalence without identity | 2/9 |
| **LOW** | `OppositeOf` | Contrastive grounding (e.g., deforestation vs. reforestation) | 2/9 |

### 5.3 Academic Citations

| KG/Tool | Key Paper | Venue |
|---------|-----------|-------|
| ConceptNet | Speer, Chin & Havasi (2017). "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." | AAAI 2017 |
| Wikidata | Vrandečić & Krötzsch (2014). "Wikidata: A Free Collaborative Knowledge Base." | CACM 57(10) |
| DBpedia | Auer et al. (2007). "DBpedia: A Nucleus for a Web of Open Data." | ISWC 2007 |
| YAGO | Suchanek, Kasneci & Weikum (2007). "YAGO: A Core of Semantic Knowledge." | WWW 2007 |
| Freebase | Bollacker et al. (2008). "Freebase: A Collaboratively Created Graph Database." | SIGMOD 2008 |
| Schema.org | Guha, Brickley & Macbeth (2016). "Schema.org: Evolution of Structured Data on the Web." | CACM 59(2) |
| SKOS | Miles & Bechhofer (2009). "SKOS Simple Knowledge Organization System Reference." | W3C Rec. |
| GraphRAG | Edge et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." | arXiv:2404.16130 |
| PharmKG | Zheng et al. (2021). "PharmKG: A Dedicated Knowledge Graph Benchmark." | Briefings in Bioinformatics |
| ENVO | Buttigieg et al. (2016). "The Environment Ontology in 2016." | J. Biomed. Semantics |
| SWEET | Raskin & Pan (2005). "Knowledge Representation in the Semantic Web for Earth and Environmental Terminology." | Comp. & Geosciences |

---

## Appendix: Relation Aliases Across KGs

| Canonical Name | ConceptNet | Wikidata | DBpedia | YAGO | SKOS | Schema.org |
|----------------|------------|----------|---------|------|------|------------|
| IsA | `/r/IsA` | P31 (instance of) | `rdf:type` | `rdf:type` | `skos:broader` (approx.) | `@type` |
| SubclassOf | — | P279 | `rdfs:subClassOf` | `rdfs:subClassOf` | `skos:broader` | — |
| PartOf | `/r/PartOf` | P361 | `dbo:isPartOf` | `schema:isPartOf` | — | `isPartOf` |
| HasPart | `/r/HasA` | P527 | `dbo:part` | `schema:hasPart` | — | `hasPart` |
| LocatedIn | `/r/AtLocation` | P131 / P276 | `dbo:location` | `schema:location` | — | `containedInPlace` |
| Causes | `/r/Causes` | P828 | — | — | — | — |
| HasEffect | — | P1542 | — | — | — | — |
| RelatedTo | `/r/RelatedTo` | — | `dbo:wikiPageWikiLink` | — | `skos:related` | `isRelatedTo` |
| SameAs | — | P460 | `owl:sameAs` | `schema:sameAs` | `skos:exactMatch` | `sameAs` |
| CreatedBy | `/r/CreatedBy` | P50 (author) | `dbo:author` | `schema:author` | — | `author` / `creator` |
| Follows | — | P155 | `dbo:predecessor` | — | — | — |
| FollowedBy | — | P156 | `dbo:successor` | — | — | `supersededBy` |
| OppositeOf | `/r/Antonym` | P461 | — | — | — | — |
| DistinctFrom | `/r/DistinctFrom` | P1889 | — | — | — | — |
| SimilarTo | `/r/SimilarTo` | — | — | — | `skos:closeMatch` | `isSimilarTo` |
| MemberOf | — | P463 | `dbo:league` etc. | `schema:memberOf` | — | `memberOf` |
| MadeOf | `/r/MadeOf` | P186 | `dbo:material` | — | — | `material` |
| UsedFor | `/r/UsedFor` | P366 | — | — | — | — |
