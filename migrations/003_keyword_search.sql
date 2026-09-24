ALTER TABLE resources
  ADD COLUMN IF NOT EXISTS search_document tsvector;

CREATE OR REPLACE FUNCTION kagglekit_resources_search_document_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.search_document :=
    setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(NEW.subtitle, '')), 'B') ||
    setweight(to_tsvector('english', array_to_string(coalesce(NEW.tags, '{}'::text[]), ' ')), 'B') ||
    setweight(to_tsvector('english', coalesce(NEW.description, '')), 'C');
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_resources_search_document ON resources;

CREATE TRIGGER trg_resources_search_document
BEFORE INSERT OR UPDATE OF title, subtitle, tags, description
ON resources
FOR EACH ROW
EXECUTE FUNCTION kagglekit_resources_search_document_update();

UPDATE resources
SET search_document =
  setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
  setweight(to_tsvector('english', coalesce(subtitle, '')), 'B') ||
  setweight(to_tsvector('english', array_to_string(coalesce(tags, '{}'::text[]), ' ')), 'B') ||
  setweight(to_tsvector('english', coalesce(description, '')), 'C');

ALTER TABLE resources
  ALTER COLUMN search_document SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_resources_search_document_gin
  ON resources USING gin(search_document);
