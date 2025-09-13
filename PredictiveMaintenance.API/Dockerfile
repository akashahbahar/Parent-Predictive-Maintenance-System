# Stage 1: Build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY PredictiveMaintenance.API/PredictiveMaintenance.API.csproj ./PredictiveMaintenance.API/
RUN dotnet restore ./PredictiveMaintenance.API/PredictiveMaintenance.API.csproj
COPY . .
COPY PredictiveMaintenance.API/model.zip ./PredictiveMaintenance.API/
WORKDIR /src/PredictiveMaintenance.API
RUN dotnet publish -c Release -o /app

# Stage 2: Runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app .
EXPOSE 8080
ENTRYPOINT ["dotnet", "PredictiveMaintenance.API.dll"]