docker run --rm -d --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres123 -e POSTGRES_DB=quotes -p 5429:5432 -v /data/quotes:/var/lib/postgresql/data postgres
