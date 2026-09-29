-- MLflow guarda sus tablas en el schema public y se conecta directo a Postgres como postgres.
-- La Data API de Supabase (roles anon y authenticated) no las necesita: se le quita todo acceso.
revoke all on schema public from anon, authenticated;
revoke all on all tables in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke all on all functions in schema public from anon, authenticated;

-- Lo mismo para las tablas que MLflow cree en futuras migraciones
alter default privileges for role postgres in schema public revoke all on tables from anon, authenticated;
alter default privileges for role postgres in schema public revoke all on sequences from anon, authenticated;
alter default privileges for role postgres in schema public revoke all on functions from anon, authenticated;

-- Función creada por MLflow: search_path fijo para que no dependa del rol que la invoque
alter function public.prevent_secrets_aad_mutation() set search_path = public, pg_temp;
