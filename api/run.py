from myapp import create_app

app = create_app()


if __name__ == '__main__':
    app = create_app()
    app.run(port=8080)


# Path: api/myapp/__init__.py