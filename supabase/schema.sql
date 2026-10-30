-- Run in the Supabase SQL editor.
-- Already created the tables? Run this instead of re-creating profiles:
--   alter table profiles alter column github drop not null, alter column portfolio drop not null,
--     alter column linkedin drop not null, alter column resume drop not null;
create table if not exists profiles (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null,
  github text,
  portfolio text,
  linkedin text,
  resume text
);

create table if not exists companies (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  website text
);

create table if not exists recipients (
  id uuid primary key default gen_random_uuid(),
  name text,
  email text not null,
  role text
);

create table if not exists emails (
  id uuid primary key default gen_random_uuid(),
  profile_id uuid not null references profiles(id),
  company_id uuid not null references companies(id),
  recipient_id uuid not null references recipients(id),
  subject text,
  body text,
  status text not null default 'draft',
  scheduled_at timestamptz,
  sent_at timestamptz,
  error text
);

create index if not exists emails_due_idx on emails (status, scheduled_at);

-- Tailored resume drafts: saved when generated, reviewed (approved/rejected) by the user, and an
-- approved draft can be used when generating a founder email.
create table if not exists resume_drafts (
  id uuid primary key default gen_random_uuid(),
  profile_id uuid not null references profiles(id),
  job_description text not null,
  draft jsonb not null,
  status text not null default 'pending_review',
  created_at timestamptz not null default now(),
  reviewed_at timestamptz
);

create index if not exists resume_drafts_profile_idx on resume_drafts (profile_id, created_at desc);

-- Which approved tailored resume an email was written from (null = the confirmed resume).
alter table emails add column if not exists resume_draft_id uuid references resume_drafts(id);

-- Only the backend (using the secret key, which bypasses RLS) should touch these tables.
alter table profiles enable row level security;
alter table companies enable row level security;
alter table recipients enable row level security;
alter table emails enable row level security;
alter table resume_drafts enable row level security;
