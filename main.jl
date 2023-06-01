using DataFrames
using CSV
using GLM
using CategoricalArrays
using StatsKit
ENV["DATAFRAMES_ROWS"] = 6

function processData(data::DataFrame)::DataFrame
    data = select(data, Not([:PassengerId, :Name]))
    data.RoomDeck = categorical([String(deck[1]) for deck in split.(data.Cabin, "/")])
    data.RoomNum = categorical([String(deck[2]) for deck in split.(data.Cabin, "/")])
    data.RoomSide = categorical([String(deck[3]) for deck in split.(data.Cabin, "/")])
    data.TotalSpent = data.RoomService + data.FoodCourt + data.ShoppingMall + data.Spa + data.VRDeck
    data = select(data, Not(:Cabin))
    data = transform(data, names(data, AbstractString) .=> categorical, renamecols=false)

    return data
end

function measureAccuracy(model, data::DataFrame)::Float64
    predictions = predict(model, processData(data))
    predictions = [
        if x < 0.5
            false
        else
            true
        end for x in predictions
    ]
    prediction_df = DataFrame(y_actual=data.Transported, y_predicted=predictions, prob_predicted=predictions)
    prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted
    accuracy = mean(prediction_df.correctly_classified)

    return accuracy

end

data = CSV.read("data/train.csv", DataFrame) |> dropmissing

fn = @formula(Transported ~ HomePlanet + CryoSleep)
logitModel = glm(fn, processData(data), Bernoulli(), LogitLink())

measureAccuracy(logitModel, data)