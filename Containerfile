# Base on alpine linux image.
FROM alpine:3.18

# Install python without apk cashe. 
RUN apk add --no-cache python3 py3-pip && \
    rm -rf /var/cache/apk/* 

# Set work directory.
WORKDIR /AskOwl

# Install python requirements for app.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy app source.
COPY src/ ./

# Create a user, change the permissions for workdir, and switch to that user.
RUN addgroup -S AskOwlUser && adduser -S AskOwlUser -G AskOwlUser
RUN chown -R AskOwlUser:AskOwlUser /AskOwl
USER AskOwlUser

# Set working port
EXPOSE 3000

# Use uvicorn on main.py with FastAPI instance 
CMD ["uvicorn", "src:app", "--host", "0.0.0.0", "--port", "3000"]
