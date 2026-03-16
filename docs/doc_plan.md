```text
docs/
├── index.md                 # Main entry point a short “readme” for the documentation.
├── api/                     # API Documentation (Interface)
│   ├── authentication.md    # e.g., how JWT works, refresh tokens
│   ├── users.md             # user-related endpoints
│   └── errors.md            # error dictionary, HTTP codes
├── architecture/            # Technical Documentation (Internal)
│   ├── overview.md          # diagrams, general overview
│   ├── database.md          # database schema, migrations
│   └── services.md          # description of business logic, services
└── operations/              # Formerly “administration” (Maintenance)
    ├── installation.md      # how to set up dev/prod environments
    ├── configuration.md     # environment variables, .env
    └── maintenance.md       # backup, logs, repair scripts
```
