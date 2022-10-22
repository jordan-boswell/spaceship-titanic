from collections import defaultdict
import pandas as pd
import unreal as u



datapath = r'C:\Users\jbos1\Desktop\Projects\Kaggle\spaceship-titanic\data\\'

# Read in the ship dataset
int_cols = ['PassengerId', 'CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported', 'Train', 'GID', 'IID', 'GroupSize', 'Num', 'Spending', 'HasSpent', 'OthersA', 'OthersB', 'OthersC', 'OthersD', 'OthersSameCabin', 'OthersNACabin', 'OthersTransported']
ship = pd.read_csv(datapath + 'ship.csv')
for int_col in int_cols:
    ship[int_col] = ship[int_col].astype('float').astype('Int64')  #this is a workaround for a known bug: https://stackoverflow.com/questions/60024262/error-converting-object-string-to-int32-typeerror-object-cannot-be-converted

# Unreal setup
#eas = u.EditorActorSubsystem()
cube_asset = u.load_asset('/Engine/BasicShapes/Cube.Cube')
sphere_asset = u.load_asset('/Engine/BasicShapes/Sphere.Sphere')
transported_material = u.load_asset('/Game/Materials/Transported.Transported')
not_transported_material = u.load_asset('/Game/Materials/NotTransported.NotTransported')

# Temp Stuff
max_cabin_size = 8
room_width = max_cabin_size * 100
room_height = 400
min_wall_width = 50
starboard_y = 100
port_y = -100
actor_width = 100
decks = ['G', 'F', 'E', 'D', 'C', 'B', 'A']  # Treat deck T separately


# Calculate the length of the ship
max_num_rooms = 0
for deck in decks:
    deck_subset = ship.loc[ship.Deck == deck, :]
    num_rooms = deck_subset.Num.max() - deck_subset.Num.min() + 1
    max_num_rooms = max(max_num_rooms, num_rooms)
ship_length = max_num_rooms * actor_width + (max_num_rooms - 1) * min_wall_width

deck_i = 0
for deck in decks:
    deck_subset = ship.loc[ship.Deck == deck, :]
    deck_min_num = deck_subset.Num.min()
    deck_max_num = deck_subset.Num.max()
    room_numbers = range(deck_min_num, deck_max_num + 1)
    z = deck_i * room_height
    deck_room_x_spacing = actor_width + (ship_length - actor_width * len(room_numbers)) / (len(room_numbers) - 1)
    for room_i in range(len(room_numbers)):
        num = room_numbers[room_i]
        x = room_i * deck_room_x_spacing
        passengers = deck_subset[deck_subset.Num == num]
        starboard_cabin_passenger_index = 0
        port_cabin_passenger_index = 0
        for index, passenger in passengers.iterrows():
            if passenger.Side == 'S':
                position = u.Vector(x, starboard_y + starboard_cabin_passenger_index * actor_width, z)
                starboard_cabin_passenger_index += 1
            else:
                position = u.Vector(x, port_y - port_cabin_passenger_index * actor_width, z)
                port_cabin_passenger_index += 1
            actor = u.EditorLevelLibrary.spawn_actor_from_object(sphere_asset, position)
            if pd.notnull(passenger.Transported):
                mesh = actor.get_component_by_class(u.StaticMeshComponent)
                mesh.set_material(0, transported_material if passenger.Transported == 1 else not_transported_material)
    deck_i += 1